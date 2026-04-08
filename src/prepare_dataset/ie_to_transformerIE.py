"""
Preprocessor modulare per Information Elements (IE) dei Wi-Fi Probe Request Burst.

SCOPO:
Convertire le IE grezze dei burst di Probe Request in feature pulite, numeriche,
robuste e adatte per modelli di machine learning (encoder tabulari, transformer).

ARCHITETTURA:
- Ogni funzione trasforma una singola IE o gruppo logico di IE
- Input: valore grezzo dal JSON del dataset
- Output: Dict[str, Any] con feature derivate numeriche
- Gestione robusta di valori mancanti (None, "None", "", ecc.)
- Nessuna eccezione non gestita: fallback a valori neutri

IE IMPLEMENTATE (140 feature totali):
  - IE0 (SSID):               4 feature
  - IE1 + IE50 (Legacy Rates): 31 feature
  - IE45 (HT Capabilities):   57 feature (AMPDU + CAP + ASEL + TXBF + MCS + RX)
  - IE107 (Interworking):      6 feature
  - IE127 (Extended Caps):    19 feature
  - IE191 (VHT Capabilities): 13 feature
  - IE221 (Vendor Specific):   9 feature

UTILIZZO:
  # Per un singolo record burst
  record_out = {}
  record_out.update(transform_ie0(...))
  record_out.update(transform_ie1_ie50_supported_rates(...))
  # ... altre IE
  
  # Oppure usando helper per dataset interi
  preprocessed = preprocess_dataset(burst_dataset)
  preprocessed = preprocess_burst(burst_record)

FEATURE CHARACTERISTICS:
  - Tipo di features: binarie (0|1), discrete (interi), continue (float)
  - Naming: <group>_<subfeat> (es. ht_cap_ldpc, legacy_rate_54mbps_supported)
  - Sempre presente: tutti i feature sono sempre emessi (mai omessi)
  - Valori mancanti: fallback a 0 o valori neutri

RIFERIMENTO SPECIFICHE:
  Vedere utils/newIEforEncoding_list.txt per dettagli sulle feature attese per ogni IE.

Autore: Preparazione dataset tesi Wi-Fi fingerprinting
"""

from typing import Any, Dict, Optional
import json
from pathlib import Path
import ast


def is_ie_missing(value: Any) -> bool:
    """
    Verifica se un valore IE è mancante o non valorizzato.
    
    Casi considerati mancanti:
    - None (valore null)
    - stringa letterale "None"
    - stringa vuota
    
    Args:
        value: Valore dell'IE da verificare
        
    Returns:
        True se il valore è mancante/non valorizzato, False altrimenti
    """
    return value is None or value == "None" or value == ""


def decode_hex_string(hex_string: str) -> str:
    """
    Decodifica una stringa esadecimale separata da ':' in una stringa ASCII.
    
    Formato atteso: "56:6f:64:61:66:6f:6e:65" (hex separato da ':')
    Risultato: "Vodafone"
    
    Args:
        hex_string: Stringa nel formato hex separato da ':' (es. "56:6f:64:61")
        
    Returns:
        Stringa decodificata in ASCII; stringa vuota se decodifica fallisce
    """
    if not hex_string or not isinstance(hex_string, str):
        return ""
    
    try:
        hex_bytes = hex_string.split(':')
        # Filtra byte vuoti e decodifica
        decoded = bytes(
            int(h, 16) for h in hex_bytes if h
        ).decode('ascii', errors='ignore')
        return decoded
    except (ValueError, AttributeError, UnicodeDecodeError):
        return ""


# ============================================================================
# Helper comuni per estrazione e conversione
# ============================================================================

def safe_int_conversion(value: Any) -> Optional[int]:
    """Converte in sicurezza un valore a intero, ritorna None se fallisce."""
    if value is None or value == "None":
        return None
    try:
        if isinstance(value, str):
            return int(value, 0)  # 0 per base automatica (10, 16 con 0x, ecc.)
        return int(value)
    except (ValueError, TypeError):
        return None


def extract_bit(value: int, bit_position: int) -> int:
    """Estrae il bit in posizione bit_position (0-indexed da destra)."""
    return (value >> bit_position) & 1


def extract_bits_range(value: int, start_bit: int, num_bits: int) -> int:
    """Estrae num_bits bit a partire da start_bit."""
    mask = (1 << num_bits) - 1
    return (value >> start_bit) & mask


def count_set_bits(value: int) -> int:
    """Conta il numero di bit a 1."""
    return bin(value).count('1')


def parse_rate_as_mbps(rate_byte: int) -> float:
    """
    Converte un rate byte 802.11 in Mbps.
    Il formato è: (rate_value & 0x7F) / 2 Mbps.
    Se il bit 7 è 1, il rate è considerato "basic rate".
    """
    rate_value = rate_byte & 0x7F
    return rate_value / 2.0


def parse_rates_from_list(rates_list: Any) -> tuple[list[float], list[int]]:
    """
    Parsare una lista di rate (possono essere int, string con lista, ecc.).
    Ritorna (list_of_rates_mbps, list_of_basic_flags).
    """
    rates = []
    basic_flags = []
    
    if not rates_list or is_ie_missing(rates_list):
        return rates, basic_flags
    
    # Se è stringa, prova a parseare come lista testuale
    if isinstance(rates_list, str):
        rates_list = rates_list.strip()
        if rates_list.startswith('[') and rates_list.endswith(']'):
            try:
                # Parsing sicuro di "[1, 2, 3, ...]"
                rates_list = ast.literal_eval(rates_list)
            except (ValueError, SyntaxError):
                return rates, basic_flags
    
    # Se è iterabile
    if isinstance(rates_list, (list, tuple)):
        for rate_byte in rates_list:
            try:
                rate_int = int(rate_byte) if isinstance(rate_byte, str) else rate_byte
                is_basic = 1 if (rate_int & 0x80) else 0
                rate_mbps = parse_rate_as_mbps(rate_int)
                rates.append(rate_mbps)
                basic_flags.append(is_basic)
            except (ValueError, TypeError):
                continue
    
    return rates, basic_flags


def extract_byte_bits(byte_value: Any) -> list[int]:
    """
    Estrae gli 8 bit di un byte come lista [bit0, bit1, ..., bit7].
    """
    if byte_value is None or is_ie_missing(byte_value):
        return [0] * 8
    
    try:
        byte_int = int(byte_value) if isinstance(byte_value, str) else byte_value
        return [extract_bit(byte_int, i) for i in range(8)]
    except (ValueError, TypeError):
        return [0] * 8


# ============================================================================
# IE1 + IE50: Legacy Supported Rates
# ============================================================================

def transform_ie1_ie50_supported_rates(
    ie1_value: Any,
    ie50_value: Any
) -> Dict[str, Any]:
    """
    Trasforma IE1 (Supported Rates) e IE50 (Extended Supported Rates) in feature.
    
    Unifica le due IE e estrae:
    - Supporto per rate standard (1, 2, 5.5, 6, ..., 54 Mbps)
    - Flag basic per ogni rate
    - Statistiche su numero, min, max, media di rate supportati
    
    Args:
        ie1_value: IE1 valore (lista di interi o stringa rappresentante lista)
        ie50_value: IE50 valore (lista di interi o stringa rappresentante lista)
    
    Returns:
        Dict con feature del tipo:
        {
            "legacy_rate_1mbps_supported": 0|1,
            "legacy_rate_2mbps_supported": 0|1,
            ...
            "legacy_rate_54mbps_supported": 0|1,
            "legacy_rate_1mbps_basic": 0|1,
            ...
            "num_supported_legacy_rates": int,
            "num_basic_legacy_rates": int,
            "min_supported_legacy_rate_mbps": float or 0,
            "max_supported_legacy_rate_mbps": float or 0,
            "mean_supported_legacy_rate_mbps": float,
            "has_only_low_legacy_rates": 0|1,
            "has_high_legacy_rate_54mbps": 0|1
        }
    """
    # Parse rate da IE1 e IE50
    rates1, basic1 = parse_rates_from_list(ie1_value)
    rates50, basic50 = parse_rates_from_list(ie50_value)
    
    # Unisci e deduplicasemi
    set_rates = set(rates1 + rates50)
    all_rates = sorted(list(set_rates))
    
    # Mappa per capire flag basic
    rate_to_basic = {}
    for r, b in zip(rates1, basic1):
        if r not in rate_to_basic:
            rate_to_basic[r] = b
    for r, b in zip(rates50, basic50):
        if r not in rate_to_basic:
            rate_to_basic[r] = b
    
    # Feature per rate standard
    standard_rates = [1.0, 2.0, 5.5, 6.0, 9.0, 11.0, 12.0, 18.0, 24.0, 36.0, 48.0, 54.0]
    result = {}
    
    for rate in standard_rates:
        is_supported = 1 if rate in set_rates else 0
        is_basic = rate_to_basic.get(rate, 0) if is_supported else 0
        
        # Nome feature
        if rate == int(rate):
            rate_name = f"{int(rate)}mbps"
        else:
            rate_name = f"{rate}mbps".replace(".", "_")
        
        result[f"legacy_rate_{rate_name}_supported"] = is_supported
        result[f"legacy_rate_{rate_name}_basic"] = is_basic
    
    # Statistiche globali
    num_supported = len(all_rates)
    num_basic = sum(1 for r in all_rates if rate_to_basic.get(r, 0))
    
    min_rate = min(all_rates) if all_rates else 0.0
    max_rate = max(all_rates) if all_rates else 0.0
    mean_rate = sum(all_rates) / len(all_rates) if all_rates else 0.0
    
    # Se solo rate bassi (< 24 Mbps)
    has_only_low = 1 if all_rates and all(r < 24.0 for r in all_rates) else 0
    has_54 = 1 if 54.0 in set_rates else 0
    
    result.update({
        "num_supported_legacy_rates": num_supported,
        "num_basic_legacy_rates": num_basic,
        "min_supported_legacy_rate_mbps": float(min_rate),
        "max_supported_legacy_rate_mbps": float(max_rate),
        "mean_supported_legacy_rate_mbps": float(mean_rate),
        "has_only_low_legacy_rates": has_only_low,
        "has_high_legacy_rate_54mbps": has_54
    })
    
    return result


def transform_ie0(ie0_value: Optional[str]) -> Dict[str, Any]:
    """
    Trasforma IE0 (SSID) in feature derivate.
    
    Feature generate:
    - ie0_is_missing: 1 se IE0 è assente/non valorizzato, 0 altrimenti (binaria)
    - ie0_is_empty: 1 se decodificato è stringa vuota, 0 altrimenti (binaria)
    - ie0_length: numero di byte della stringa hex (continua)
    - ie0_num_tokens: numero di parole (token) nella stringa decodificata (discreta)
    
    Args:
        ie0_value: Valore IE0 dal dataset (stringa hex, "None", None, o "")
        
    Returns:
        Dictionary con i feature estratti:
        {
            "ie0_is_missing": int,   # 0 o 1
            "ie0_is_empty": int,     # 0 o 1
            "ie0_length": int,       # numero byte
            "ie0_num_tokens": int    # numero token
        }
        
    Examples:
        >>> transform_ie0("56:6f:64:61:66:6f:6e:65:57:69:46:69")
        {'ie0_is_missing': 0, 'ie0_is_empty': 0, 'ie0_length': 12, 'ie0_num_tokens': 1}
        
        >>> transform_ie0("57:69:6e:64:33:20:48:55:42:2d:36:44:31:36:31:39")
        {'ie0_is_missing': 0, 'ie0_is_empty': 0, 'ie0_length': 16, 'ie0_num_tokens': 2}
        
        >>> transform_ie0(None)
        {'ie0_is_missing': 1, 'ie0_is_empty': 0, 'ie0_length': 0, 'ie0_num_tokens': 0}
    """
    # Verifica se IE0 è mancante
    if is_ie_missing(ie0_value):
        return {
            "ie0_is_missing": 1,
            "ie0_is_empty": 0,
            "ie0_length": 0,
            "ie0_num_tokens": 0
        }
    
    # IE0 è valorizzato: calcola feature derivate
    # Gestisci token SSID separati da '-'
    ssid_tokens = ie0_value.split('-') if '-' in ie0_value else [ie0_value]
    
    total_length = 0
    all_decoded = []
    for token in ssid_tokens:
        hex_bytes = token.split(':')
        total_length += len(hex_bytes)
        decoded = decode_hex_string(token)
        if decoded:
            all_decoded.append(decoded)
    
    # ie0_length = totale byte in tutti i token
    ie0_length = total_length
    
    # ie0_num_tokens = numero di token SSID (non parole nella stringa decodificata)
    ie0_num_tokens = len(ssid_tokens)
    
    # ie0_is_empty = 1 se tutti i token decodificati sono vuoti o whitespace
    is_empty = 1 if not any(d.strip() for d in all_decoded) else 0
    
    return {
        "ie0_is_missing": 0,
        "ie0_is_empty": is_empty,
        "ie0_length": ie0_length,
        "ie0_num_tokens": ie0_num_tokens
    }


# ============================================================================
# IE45: High Throughput (802.11n) Capabilities
# ============================================================================

def transform_ie45_ampduparam(ampdu_param: Any) -> Dict[str, Any]:
    """Trasforma IE45 A-MPDU Parameters."""
    ampdu_int = safe_int_conversion(ampdu_param) or 0
    max_ampdu_exp = extract_bits_range(ampdu_int, 0, 2)
    min_spacing = extract_bits_range(ampdu_int, 2, 3)
    max_ampdu_bytes = (1 << (13 + max_ampdu_exp)) - 1
    return {
        "ht_max_ampdu_length_exponent": int(max_ampdu_exp),
        "ht_min_mpdu_start_spacing": int(min_spacing),
        "ht_max_ampdu_length_bytes": int(max_ampdu_bytes)
    }


def transform_ie45_capabilities(ht_cap: Any) -> Dict[str, Any]:
    """Trasforma IE45 HT Capabilities in feature binarie."""
    ht_cap_int = safe_int_conversion(ht_cap) or 0
    result = {
        "ht_cap_ldpc": int(extract_bit(ht_cap_int, 0)),
        "ht_cap_supported_channel_width": int(extract_bit(ht_cap_int, 1)),
        "ht_cap_sm_power_save": int(extract_bits_range(ht_cap_int, 2, 2)),
        "ht_cap_greenfield": int(extract_bit(ht_cap_int, 4)),
        "ht_cap_short_gi_20": int(extract_bit(ht_cap_int, 5)),
        "ht_cap_short_gi_40": int(extract_bit(ht_cap_int, 6)),
        "ht_cap_tx_stbc": int(extract_bit(ht_cap_int, 7)),
        "ht_cap_rx_stbc": int(extract_bits_range(ht_cap_int, 8, 2)),
        "ht_cap_delayed_block_ack": int(extract_bit(ht_cap_int, 10)),
        "ht_cap_max_amsdu_length": int(extract_bit(ht_cap_int, 11)),
        "ht_cap_dsss_cck_40": int(extract_bit(ht_cap_int, 12)),
        "ht_cap_40mhz_intolerant": int(extract_bit(ht_cap_int, 13)),
        "ht_cap_lsig_txop_protection": int(extract_bit(ht_cap_int, 14)),
        "ht_cap_num_enabled_flags": int(count_set_bits(ht_cap_int))
    }
    return result


def transform_ie45_asel(asel_value: Any) -> Dict[str, Any]:
    """Trasforma IE45 ASEL (Antenna Selection)."""
    asel_int = safe_int_conversion(asel_value) or 0
    return {
        "ht_asel_supported": int(extract_bit(asel_int, 0)),
        "ht_asel_explicit_csi_feedback_based_tx_asel": int(extract_bit(asel_int, 1)),
        "ht_asel_antenna_indices_feedback_based_tx_asel": int(extract_bit(asel_int, 2)),
        "ht_asel_explicit_csi_feedback": int(extract_bit(asel_int, 3)),
        "ht_asel_antenna_indices_feedback": int(extract_bit(asel_int, 4)),
        "ht_asel_rx_asel": int(extract_bit(asel_int, 5)),
        "ht_asel_tx_sounding_ppdu": int(extract_bit(asel_int, 6)),
        "ht_asel_num_enabled_flags": int(count_set_bits(asel_int))
    }


def transform_ie45_txbf(txbf_value: Any) -> Dict[str, Any]:
    """Trasforma IE45 Transmit Beamforming Capabilities."""
    txbf_int = safe_int_conversion(txbf_value) or 0
    return {
        "ht_txbf_implicit_rx": int(extract_bit(txbf_int, 0)),
        "ht_txbf_rx_staggered_sounding": int(extract_bit(txbf_int, 1)),
        "ht_txbf_tx_staggered_sounding": int(extract_bit(txbf_int, 2)),
        "ht_txbf_rx_ndp": int(extract_bit(txbf_int, 3)),
        "ht_txbf_tx_ndp": int(extract_bit(txbf_int, 4)),
        "ht_txbf_implicit_tx": int(extract_bit(txbf_int, 5)),
        "ht_txbf_calibration": int(extract_bits_range(txbf_int, 6, 2)),
        "ht_txbf_csi_feedback": int(extract_bits_range(txbf_int, 8, 2)),
        "ht_txbf_noncompressed_feedback": int(extract_bits_range(txbf_int, 10, 2)),
        "ht_txbf_compressed_feedback": int(extract_bits_range(txbf_int, 12, 2)),
        "ht_txbf_grouping": int(extract_bits_range(txbf_int, 14, 2)),
        "ht_txbf_num_enabled_flags": int(count_set_bits(txbf_int))
    }


def transform_ie45_mcsset_summary(tx_unequal_mod: Any, tx_rx_mcs_not_equal: Any, tx_max_ss: Any, tx_set_defined: Any, highest_data_rate: Any) -> Dict[str, Any]:
    """Trasforma IE45 MCS Set Summary."""
    tx_un = safe_int_conversion(tx_unequal_mod) or 0
    tx_diff = safe_int_conversion(tx_rx_mcs_not_equal) or 0
    tx_ss = safe_int_conversion(tx_max_ss) or 0
    tx_def = safe_int_conversion(tx_set_defined) or 0
    hdr = safe_int_conversion(highest_data_rate) or 0
    
    # ht_mcs_summary_present = 1 se almeno un campo MCS è presente
    mcs_present = (not is_ie_missing(tx_unequal_mod) or 
                   not is_ie_missing(tx_rx_mcs_not_equal) or 
                   not is_ie_missing(tx_max_ss) or 
                   not is_ie_missing(tx_set_defined) or 
                   not is_ie_missing(highest_data_rate))
    
    return {
        "ht_tx_unequal_modulation_supported": int(tx_un),
        "ht_tx_rx_mcs_sets_differ": int(tx_diff),
        "ht_tx_max_spatial_streams": int(tx_ss),
        "ht_tx_mcs_set_defined": int(tx_def),
        "ht_rx_highest_supported_data_rate_mbps": float(hdr),
        "ht_has_multiple_spatial_streams": 1 if int(tx_ss) > 0 else 0,
        "ht_highest_rate_is_zero": 1 if int(hdr) == 0 else 0,
        "ht_mcs_summary_present": 1 if mcs_present else 0
    }


def transform_ie45_rx_mcs_bitmask(bm_0_7: Any, bm_8_15: Any, bm_16_23: Any, bm_24_31: Any, bm_32: Any, bm_33_38: Any, bm_39_52: Any, bm_53_76: Any) -> Dict[str, Any]:
    """Trasforma IE45 RX MCS Bitmask in feature sintetiche."""
    full_bitmask = 0
    for mask_val, offset in [(bm_0_7, 0), (bm_8_15, 8), (bm_16_23, 16), (bm_24_31, 24), (bm_32, 32), (bm_33_38, 33), (bm_39_52, 39), (bm_53_76, 53)]:
        mask_int = safe_int_conversion(mask_val)
        if mask_int is not None:
            full_bitmask |= (mask_int << offset)
    
    supported = [i for i in range(77) if (full_bitmask >> i) & 1]
    num_sup = len(supported)
    lo = supported[0] if supported else 0
    hi = supported[-1] if supported else 0
    
    # Continuità
    cont_zero = 1 if (supported and all((full_bitmask >> i) & 1 for i in range(hi + 1))) else 0
    
    # Gap count
    gaps = 0
    prev = 0
    in_gap = False
    for i in range(77):
        curr = (full_bitmask >> i) & 1
        if prev == 1 and curr == 0:
            in_gap = True
        elif prev == 0 and curr == 1 and in_gap:
            gaps += 1
            in_gap = False
        prev = curr
    
    return {
        "ht_rx_mcs_num_supported": int(num_sup),
        "ht_rx_mcs_lowest_supported_index": int(lo),
        "ht_rx_mcs_highest_supported_index": int(hi),
        "ht_rx_mcs_supports_mcs0": int((full_bitmask & 1)),
        "ht_rx_mcs_supports_mcs7": int(((full_bitmask >> 7) & 1)),
        "ht_rx_mcs_supports_mcs15": int(((full_bitmask >> 15) & 1)),
        "ht_rx_mcs_supports_mcs23": int(((full_bitmask >> 23) & 1)),
        "ht_rx_mcs_supports_mcs31": int(((full_bitmask >> 31) & 1)),
        "ht_rx_mcs_supports_any_above_31": int(1 if (full_bitmask >> 32) else 0),
        "ht_rx_mcs_supports_any_above_76": int(1 if (full_bitmask >> 77) else 0),
        "ht_rx_mcs_contiguous_from_zero": int(cont_zero),
        "ht_rx_mcs_num_gaps": int(gaps)
    }


# ============================================================================
# IE107: Interworking
# ============================================================================

def transform_ie107_interworking(access_net_type: Any, internet: Any = None, asra: Any = None, esr: Any = None, uesa: Any = None) -> Dict[str, Any]:
    """Trasforma IE107 Interworking."""
    ant = safe_int_conversion(access_net_type) or 0
    net = safe_int_conversion(internet) or 0
    a = safe_int_conversion(asra) or 0
    e = safe_int_conversion(esr) or 0
    u = safe_int_conversion(uesa) or 0
    
    # interworking_present = 1 solo se almeno un campo IE107 è presente
    present = (not is_ie_missing(access_net_type) or 
               not is_ie_missing(internet) or 
               not is_ie_missing(asra) or 
               not is_ie_missing(esr) or 
               not is_ie_missing(uesa))
    
    return {
        "interworking_access_network_type": int(ant),
        "interworking_internet": int(net),
        "interworking_asra": int(a),
        "interworking_esr": int(e),
        "interworking_uesa": int(u),
        "interworking_present": 1 if present else 0
    }


def parse_serialized_list(value: Any, max_length: int = 8) -> list[int]:
    """
    Parsa una stringa serializzata come lista (es. "[1, 2, 3]") o lista/tupla diretta.
    Ritorna lista di int, clampando valori >255 a 255, padding/truncating a max_length.
    Se mancante o invalida, ritorna [0] * max_length.
    """
    if is_ie_missing(value):
        return [0] * max_length
    
    try:
        if isinstance(value, str):
            if value.startswith('[') or value.startswith('('):
                parsed = ast.literal_eval(value)
                if isinstance(parsed, (list, tuple)):
                    # Clamp valori >255 e converti a int
                    parsed = [min(int(x), 255) for x in parsed if isinstance(x, (int, float))]
                else:
                    return [0] * max_length
            else:
                return [0] * max_length
        elif isinstance(value, (list, tuple)):
            parsed = [min(int(x), 255) for x in value if isinstance(x, (int, float))]
        else:
            return [0] * max_length
        
        # Pad o truncate a max_length
        if len(parsed) < max_length:
            parsed.extend([0] * (max_length - len(parsed)))
        elif len(parsed) > max_length:
            parsed = parsed[:max_length]
        
        return parsed
    except (ValueError, SyntaxError, TypeError):
        return [0] * max_length


# ============================================================================
# IE127: Extended Capabilities
# ============================================================================

def transform_ie127_extended_capabilities(ie127_0: Any, ie127_1: Any = None) -> Dict[str, Any]:
    """Trasforma IE127 Extended Capabilities bytes in feature binarie."""
    b0_bytes = parse_serialized_list(ie127_0, 8)
    b1_bytes = parse_serialized_list(ie127_1, 8)
    
    # Estrai bit da ciascun byte
    b0_bits = []
    for byte in b0_bytes:
        b0_bits.extend([extract_bit(byte, i) for i in range(8)])
    
    b1_bits = []
    for byte in b1_bytes:
        b1_bits.extend([extract_bit(byte, i) for i in range(8)])
    
    result = {f"extcap_byte0_bit{i}": b0_bits[i] for i in range(8)}
    result.update({f"extcap_byte1_bit{i}": b1_bits[i] for i in range(8)})
    result["extcap_byte0_num_enabled_bits"] = sum(b0_bits[:8])
    result["extcap_byte1_num_enabled_bits"] = sum(b1_bits[:8])
    
    # extcap_present_any = 1 se almeno un byte è presente (non tutti zero)
    has_data = any(b0_bytes) or any(b1_bytes)
    result["extcap_present_any"] = 1 if has_data else 0
    
    return result


# ============================================================================
# IE191: VHT Capabilities
# ============================================================================

def transform_ie191_vht_capabilities(vht_cap: Any) -> Dict[str, Any]:
    """Trasforma IE191 VHT Capabilities."""
    if is_ie_missing(vht_cap):
        return {
            "vht_present": 0,
            "vht_max_mpdu_length": 0,
            "vht_supported_channel_width_set": 0,
            "vht_rx_ldpc": 0,
            "vht_short_gi_80": 0,
            "vht_short_gi_160": 0,
            "vht_tx_stbc": 0,
            "vht_rx_stbc": 0,
            "vht_su_beamformer": 0,
            "vht_su_beamformee": 0,
            "vht_mu_beamformer": 0,
            "vht_mu_beamformee": 0,
            "vht_num_enabled_flags": 0
        }
    
    vht_int = safe_int_conversion(vht_cap) or 0
    return {
        "vht_present": 1,
        "vht_max_mpdu_length": int(extract_bits_range(vht_int, 0, 2)),
        "vht_supported_channel_width_set": int(extract_bits_range(vht_int, 2, 2)),
        "vht_rx_ldpc": int(extract_bit(vht_int, 4)),
        "vht_short_gi_80": int(extract_bit(vht_int, 5)),
        "vht_short_gi_160": int(extract_bit(vht_int, 6)),
        "vht_tx_stbc": int(extract_bit(vht_int, 7)),
        "vht_rx_stbc": int(extract_bits_range(vht_int, 8, 3)),
        "vht_su_beamformer": int(extract_bit(vht_int, 11)),
        "vht_su_beamformee": int(extract_bit(vht_int, 12)),
        "vht_mu_beamformer": int(extract_bit(vht_int, 13)),
        "vht_mu_beamformee": int(extract_bit(vht_int, 14)),
        "vht_num_enabled_flags": int(count_set_bits(vht_int))
    }


# ============================================================================
# IE221: Vendor Specific
# ============================================================================

def transform_ie221_vendor_specific(oui_list: Any, type_list: Any = None) -> Dict[str, Any]:
    """Trasforma IE221 Vendor Specific in feature sintetiche."""
    # Parse OUI e type list
    ouis = []
    types = []
    
    def parse_value(value: Any) -> list:
        if is_ie_missing(value):
            return []
        if isinstance(value, (list, tuple)):
            return list(value)
        if isinstance(value, str):
            if value.startswith('[') or value.startswith('('):
                try:
                    parsed = ast.literal_eval(value)
                    return list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]
                except (ValueError, SyntaxError):
                    return []
            else:
                # Stringa scalare, prova a convertirla in int
                try:
                    return [int(value)]
                except ValueError:
                    return []
        if isinstance(value, int):
            return [value]
        return []
    
    ouis = parse_value(oui_list)
    types = parse_value(type_list) if type_list is not None else []
    
    # Known vendors (OUI values)
    vendor_map = {
        20722: 'microsoft',       # 000427
        8754: 'broadcom',          # 00222f
        16887: 'apple',            # 004096
        20306: 'samsung',          # 004fd9
        26970: 'qualcomm'          # 0006b9
    }
    
    vendor_count = len(set(ouis)) if ouis else 0
    type_count = len(set(types)) if types else 0
    
    result = {
        "vendor_ie_count": len(ouis),
        "vendor_unique_oui_count": vendor_count,
        "vendor_unique_type_count": type_count
    }
    
    # Known vendor flags
    for oui in (ouis or []):
        oui_int = safe_int_conversion(oui) or 0
        for oui_val, vendor_name in vendor_map.items():
            if oui_int == oui_val:
                result[f"vendor_has_{vendor_name}_oui"] = 1
                break
    
    # Set default flags if not already set
    for vendor in ['microsoft', 'broadcom', 'apple', 'samsung', 'qualcomm']:
        if f"vendor_has_{vendor}_oui" not in result:
            result[f"vendor_has_{vendor}_oui"] = 0
    
    # Conta oui unici noti
    known_oui_count = sum(1 for oui in set(ouis) if safe_int_conversion(oui) in vendor_map)
    result["vendor_has_unknown_oui"] = 1 if vendor_count > known_oui_count else 0
    
    return result


def preprocess_burst(burst_record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocessa un singolo record di burst estraendo SOLO le feature IE trasformate.
    
    Applica tutte le trasformazioni IE disponibili e restituisce SOLO i nuovi feature
    (non include i campi originali del record).
    
    Args:
        burst_record: Record originale del burst con tutte le IE
        
    Returns:
        Dict con SOLO le feature IE trasformate (140 feature totali)
    """
    result = {}
    
    # IE0 (SSID)
    result.update(transform_ie0(burst_record.get("ie0")))
    
    # IE1 + IE50 (Legacy Rates)
    result.update(transform_ie1_ie50_supported_rates(
        burst_record.get("ie1"),
        burst_record.get("ie50")
    ))
    
    # IE45 A-MPDU Parameters
    result.update(transform_ie45_ampduparam(burst_record.get("ie45_ampduparam")))
    
    # IE45 Capabilities
    result.update(transform_ie45_capabilities(burst_record.get("ie45_capabilities")))
    
    # IE45 ASEL
    result.update(transform_ie45_asel(burst_record.get("ie45_asel")))
    
    # IE45 TXBF
    result.update(transform_ie45_txbf(burst_record.get("ie45_txbf")))
    
    # IE45 MCS Set Summary
    result.update(transform_ie45_mcsset_summary(
        burst_record.get("ie45_mcsset_txunequalmod"),
        burst_record.get("ie45_mcsset_txrxmcsnotequal"),
        burst_record.get("ie45_mcsset_txmaxss"),
        burst_record.get("ie45_mcsset_txsetdefined"),
        burst_record.get("ie45_mcsset_highestdatarate")
    ))
    
    # IE45 RX MCS Bitmask
    result.update(transform_ie45_rx_mcs_bitmask(
        burst_record.get("ie45_rxbitmask_0to7"),
        burst_record.get("ie45_rxbitmask_8to15"),
        burst_record.get("ie45_rxbitmask_16to23"),
        burst_record.get("ie45_rxbitmask_24to31"),
        burst_record.get("ie45_rxbitmask_32"),
        burst_record.get("ie45_rxbitmask_33to38"),
        burst_record.get("ie45_rxbitmask_39to52"),
        burst_record.get("ie45_rxbitmask_53to76")
    ))
    
    # IE107 Interworking
    result.update(transform_ie107_interworking(
        burst_record.get("ie107_access_network_type"),
        burst_record.get("ie107_internet"),
        burst_record.get("ie107_asra"),
        burst_record.get("ie107_esr"),
        burst_record.get("ie107_uesa")
    ))
    
    # IE127 Extended Capabilities
    result.update(transform_ie127_extended_capabilities(
        burst_record.get("ie127_0"),
        burst_record.get("ie127_1")
    ))
    
    # IE191 VHT Capabilities
    result.update(transform_ie191_vht_capabilities(burst_record.get("ie191")))
    
    # IE221 Vendor Specific
    oui_list = []
    type_list = []
    for i in range(10):  # 0 to 9
        oui_key = f"ie221_oui_{i}"
        type_key = f"ie221_type_{i}"
        oui_val = burst_record.get(oui_key)
        type_val = burst_record.get(type_key)
        if not is_ie_missing(oui_val) and not is_ie_missing(type_val):
            oui_list.append(oui_val)
            type_list.append(type_val)
    result.update(transform_ie221_vendor_specific(oui_list, type_list))
    
    # Preserve the original label
    result["label"] = burst_record.get("label")
    
    return result


def preprocess_dataset(burst_dataset: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Preprocessa un intero dataset di burst aggiungendo feature IE trasformate.
    
    Args:
        burst_dataset: Dictionary con struttura {burst_id: record}
        
    Returns:
        Dictionary preprocessato con feature IE aggiunte a ogni burst
    """
    burst_ids = list(burst_dataset.keys())
    burst_records = list(burst_dataset.values())
    processed_records = preprocess_list(burst_records)
    return dict(zip(burst_ids, processed_records))


def preprocess_list(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Preprocessa una lista di record burst, applicando le trasformazioni IE a ciascuno.
    
    Utile per integrare il preprocessing con dataset Python (classi Dataset, liste, ecc.)
    dove i record sono già separati da label, mac_addresses, ecc. (come in ProbeDataset).
    
    Ogni record viene preprocessato indipendentemente senza dipendenza da indici o ID.
    Il campo 'label' viene preservato se presente nel record originale.
    
    Args:
        records: Lista di dizionari, ciascuno rappresentante un record burst grezzo.
                 Ogni record può contenere IE grezze e opzionalmente un campo 'label'.
        
    Returns:
        Lista di dizionari preprocessati con feature IE trasformate (140 feature totali).
        
    Raises:
        TypeError: Se records non è una lista, oppure se un elemento non è un dizionario.
        
    Example:
        >>> raw_records = [
        ...     {"ie0": "56:6f:64:61:66:6f:6e:65", "ie1": "[130, 132, 139]", "label": 0},
        ...     {"ie0": None, "ie1": "[140, 148]", "label": 1},
        ...     {"ie0": "", "ie50": "[48, 72]"}
        ... ]
        >>> preprocessed = preprocess_list(raw_records)
        >>> len(preprocessed)
        3
        >>> "ie0_is_missing" in preprocessed[0]
        True
        >>> preprocessed[0]["label"]
        0
    """
    if not isinstance(records, list):
        raise TypeError(f"Expected list, got {type(records).__name__}")
    
    preprocessed = []
    for i, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(
                f"Expected dict at index {i}, got {type(record).__name__}"
            )
        preprocessed.append(preprocess_burst(record))
    
    return preprocessed


# ============================================================================
# Bulk Preprocessing: JSON → JSON (Conversion)
# ============================================================================

def preprocess_json_file(
    input_json_path: str,
    output_json_path: str,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Preprocessing di un singolo file JSON.
    
    Legge il JSON originale, preprocessa ogni record e salva un nuovo file JSON.
    
    Args:
        input_json_path: Percorso al file JSON sorgente
        output_json_path: Percorso dove salvare il file JSON preprocessato
        verbose: Se True, stampa log di progresso
    
    Returns:
        Dict con statistiche preprocessing
    """
    input_path = Path(input_json_path)
    output_path = Path(output_json_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_json_path}")
    
    if verbose:
        print(f"Preprocessing JSON: {input_json_path} -> {output_json_path}")
    
    # Leggi il file JSON originale
    with open(input_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    if not isinstance(original_data, dict):
        raise ValueError(f"Expected JSON object (dict), got {type(original_data).__name__}")
    
    num_records = len(original_data)
    if verbose:
        print(f"Loaded {num_records} records")
    
    # Preprocessa ogni record usando preprocess_list
    burst_ids = list(original_data.keys())
    burst_records = list(original_data.values())
    
    processed_records = preprocess_list(burst_records)
    
    # Ricostituisci dictionary con burst_id come chiave
    processed_data = dict(zip(burst_ids, processed_records))
    
    # Crea cartella di output se non esiste
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Salva il nuovo file JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=4)
    
    if verbose:
        output_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"Saved {output_size_mb:.2f} MB to {output_json_path}")
    
    stats = {
        "input_file": str(input_json_path),
        "output_file": str(output_json_path),
        "num_records_processed": num_records,
        "output_file_exists": output_path.exists()
    }
    
    return stats




