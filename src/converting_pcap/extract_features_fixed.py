# script by lucia pintor
# modified: defensive parsing for UJI/real PCAPs and PyShark edge cases
import os

import pyshark as ps
from datetime import datetime
import json
try:
    from .convert_hexadecimal_fixed import convert_hexadecimal, convert_hexadecimal_list
except:
    from convert_hexadecimal_fixed import convert_hexadecimal, convert_hexadecimal_list


def extract_from_pcap(pcap_file, max_packets=None):
    """
    Extract features from a PCAP file and return a list of packets.

    If max_packets is not None, only the first max_packets packets are parsed.
    """

    packet_list_summary = []
    packet_list = None
    skipped_packets = 0

    try:
        packet_list = ps.FileCapture(input_file=pcap_file, use_json=True)

        for i, pkt in enumerate(packet_list):
            if max_packets is not None and i >= max_packets:
                break

            try:
                pkt_summary = {}
                pkt_summary["timestamp"] = extract_timestamp(pcap_frame=pkt)
                pkt_summary["mac"] = extract_mac(pcap_frame=pkt)
                pkt_summary["seq"] = extract_sequence_number(pcap_frame=pkt)
                pkt_summary.update(extract_tag_paramenters(pcap_frame=pkt))
                packet_list_summary.append(pkt_summary)

            except Exception as packet_error:
                skipped_packets += 1
                print(f"Warning: skipped packet because of parsing error: {packet_error}")

    finally:
        if packet_list is not None:
            try:
                packet_list.close()
            except Exception as close_error:
                print(f"Warning: PyShark close failed: {close_error}")

    if skipped_packets > 0:
        print(f"Warning: skipped packets: {skipped_packets}")

    return packet_list_summary

def extract_timestamp(pcap_frame):
    """
    Extract timestamp supporting both numeric and ISO PyShark formats.

    Examples:
    - 1700000000.123456
    - 2023-02-28T10:06:13.608248000Z
    """

    timestamp_str = str(pcap_frame.sniff_timestamp)

    # Case 1: numeric timestamp.
    try:
        timestamp_datetime = datetime.fromtimestamp(float(timestamp_str))
        return timestamp_datetime.strftime("%Y-%m-%dT%H:%M:%S.%f")
    except (TypeError, ValueError):
        pass

    # Case 2: ISO timestamp ending with Z.
    if timestamp_str.endswith("Z"):
        timestamp_str = timestamp_str[:-1]

    # Python datetime supports up to 6 fractional digits. Some PCAPs expose
    # nanoseconds, e.g. 608248000, so we trim to microseconds.
    if "." in timestamp_str:
        date_part, frac_part = timestamp_str.split(".", 1)
        frac_part = (frac_part + "000000")[:6]
        timestamp_str = f"{date_part}.{frac_part}"

    try:
        timestamp_datetime = datetime.fromisoformat(timestamp_str)
        return timestamp_datetime.strftime("%Y-%m-%dT%H:%M:%S.%f")
    except ValueError:
        # Last fallback: keep the original timestamp string rather than crashing.
        return str(pcap_frame.sniff_timestamp)


def extract_mac(pcap_frame):
    return str(pcap_frame.wlan.sa)


def extract_sequence_number(pcap_frame):
    try:
        return int(pcap_frame.wlan.seq)
    except Exception:
        return -1


def extract_wlan_mgt_layer(pcap_frame):
    return pcap_frame.layers[3]


def extract_tag_paramenters(pcap_frame):
    """
    Extract Information Elements from the WLAN management layer.
    """

    ie_dict = {}

    try:
        tag_list = extract_wlan_mgt_layer(pcap_frame).all.tag
    except Exception:
        return ie_dict

    index_221 = 0
    index_127 = 0

    for t in tag_list:
        try:
            ie_info = extract_ie(t)
        except Exception as error:
            tag_number = getattr(t, "number", "unknown")
            ie_info = {f"ie{tag_number}_present": 1}
            print(f"Warning: could not fully parse IE {tag_number}: {error}")

        ie_info_keys = list(ie_info.keys())
        if not ie_info_keys:
            continue

        # IE 221 can have multiple instances in the same packet.
        if "221_oui" in ie_info_keys[0]:
            for ie_key in ie_info_keys:
                ie_dict[f"{ie_key}_{index_221}"] = ie_info[ie_key]
            index_221 += 1

        # IE 127 may also appear in repeated parsed structures.
        elif "127" in ie_info_keys[0]:
            for ie_key in ie_info_keys:
                ie_dict[f"{ie_key}_{index_127}"] = ie_info[ie_key]
            index_127 += 1

        else:
            add_ie_info_without_overwrite(ie_dict, ie_info)

    return ie_dict


def add_ie_info_without_overwrite(ie_dict, ie_info):
    """
    Add IE fields to the packet dictionary without overwriting duplicates.

    If the same IE key appears multiple times, it is saved as:
    ie1, ie1_1, ie1_2, ...
    """

    for ie_key, value in ie_info.items():
        if ie_key not in ie_dict:
            ie_dict[ie_key] = value
            continue

        duplicate_index = 1
        new_key = f"{ie_key}_{duplicate_index}"

        while new_key in ie_dict:
            duplicate_index += 1
            new_key = f"{ie_key}_{duplicate_index}"

        ie_dict[new_key] = value


def extract_ie(tag_param):
    tag_number = int(tag_param.number)

    if tag_number == 0:
        return extract_ie0_value(tag_param)
    elif tag_number == 1:
        return extract_ie1_value(tag_param)
    elif tag_number == 3:
        return extract_ie3_value(tag_param)
    elif tag_number == 45:
        return extract_ie45_value(tag_param)
    elif tag_number == 50:
        return extract_ie50_value(tag_param)
    elif tag_number == 221:
        return extract_ie221_value(tag_param)
    elif tag_number == 127:
        return extract_ie127_value(tag_param)
    elif tag_number == 107:
        return extract_ie107_value(tag_param)
    elif tag_number == 191:
        return extract_ie191_value(tag_param)
    else:
        # For unsupported IEs, keep a presence flag instead of silently losing it.
        return {f"{get_ie_id(tag_param)}_present": 1}


def get_ie_id(tag_param):
    return "ie{}".format(tag_param.number)


def safe_get(obj, attr, default=0):
    try:
        return getattr(obj, attr)
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def join_and_convert(value):
    try:
        return convert_hexadecimal("".join(value))
    except Exception:
        return convert_hexadecimal(value)


def get_rxbitmask_value(rxbitmask, key, default=0):
    try:
        return convert_hexadecimal(rxbitmask.get(key, default))
    except Exception:
        return default


def extract_ie0_value(tag_param):
    """
    Information Element 0: SSID.
    """
    return {get_ie_id(tag_param): str(safe_get(tag_param, "ssid", ""))}


def extract_ie1_value(tag_param):
    """
    Information Element 1: Supported Rates.
    """
    return {get_ie_id(tag_param): convert_hexadecimal_list(safe_get(tag_param, "supported_rates", []))}


def extract_ie3_value(tag_param):
    """
    Information Element 3: Current Channel.
    """
    return {get_ie_id(tag_param): convert_hexadecimal(safe_get(tag_param, "current_channel", 0))}


def extract_ie45_value(tag_param):
    """
    Information Element 45: HT Capabilities.
    """

    ampduparam = join_and_convert(safe_get(tag_param, "ampduparam", 0))
    asel = join_and_convert(safe_get(tag_param, "asel", 0))
    capabilities = join_and_convert(safe_get(tag_param, "capabilities", 0))
    txbf = join_and_convert(safe_get(tag_param, "txbf", 0))

    mcsset = safe_get(tag_param, "mcsset", None)

    mcsset_txunequalmod = safe_int(safe_get(mcsset, "txunequalmod", 0))
    mcsset_txrxmcsnotequal = safe_int(safe_get(mcsset, "txrxmcsnotequal", 0))
    mcsset_txmaxss = convert_hexadecimal(safe_get(mcsset, "txmaxss", 0))
    mcsset_txsetdefined = safe_int(safe_get(mcsset, "txsetdefined", 0))
    mcsset_highestdatarate = convert_hexadecimal(safe_get(mcsset, "highestdatarate", 0))

    rxbitmask = safe_get(mcsset, "rxbitmask", {})

    rxbitmask_0to7 = get_rxbitmask_value(rxbitmask, "0to7")
    rxbitmask_8to15 = get_rxbitmask_value(rxbitmask, "8to15")
    rxbitmask_16to23 = get_rxbitmask_value(rxbitmask, "16to23")
    rxbitmask_24to31 = get_rxbitmask_value(rxbitmask, "24to31")
    rxbitmask_32 = get_rxbitmask_value(rxbitmask, "32")
    rxbitmask_33to38 = get_rxbitmask_value(rxbitmask, "33to38")
    rxbitmask_39to52 = get_rxbitmask_value(rxbitmask, "39to52")
    rxbitmask_53to76 = get_rxbitmask_value(rxbitmask, "53to76")

    return {
        "{}_ampduparam".format(get_ie_id(tag_param)): ampduparam,
        "{}_asel".format(get_ie_id(tag_param)): asel,
        "{}_capabilities".format(get_ie_id(tag_param)): capabilities,
        "{}_txbf".format(get_ie_id(tag_param)): txbf,
        "{}_mcsset_txunequalmod".format(get_ie_id(tag_param)): mcsset_txunequalmod,
        "{}_mcsset_txrxmcsnotequal".format(get_ie_id(tag_param)): mcsset_txrxmcsnotequal,
        "{}_mcsset_txmaxss".format(get_ie_id(tag_param)): mcsset_txmaxss,
        "{}_mcsset_txsetdefined".format(get_ie_id(tag_param)): mcsset_txsetdefined,
        "{}_mcsset_highestdatarate".format(get_ie_id(tag_param)): mcsset_highestdatarate,
        "{}_rxbitmask_0to7".format(get_ie_id(tag_param)): rxbitmask_0to7,
        "{}_rxbitmask_8to15".format(get_ie_id(tag_param)): rxbitmask_8to15,
        "{}_rxbitmask_16to23".format(get_ie_id(tag_param)): rxbitmask_16to23,
        "{}_rxbitmask_24to31".format(get_ie_id(tag_param)): rxbitmask_24to31,
        "{}_rxbitmask_32".format(get_ie_id(tag_param)): rxbitmask_32,
        "{}_rxbitmask_33to38".format(get_ie_id(tag_param)): rxbitmask_33to38,
        "{}_rxbitmask_39to52".format(get_ie_id(tag_param)): rxbitmask_39to52,
        "{}_rxbitmask_53to76".format(get_ie_id(tag_param)): rxbitmask_53to76,
    }


def extract_ie50_value(tag_param):
    """
    Information Element 50: Extended Supported Rates.
    """
    return {get_ie_id(tag_param): convert_hexadecimal_list(safe_get(tag_param, "extended_supported_rates", []))}


def extract_ie221_value(tag_param):
    """
    Information Element 221: Vendor Specific.
    """
    return {
        "{}_oui".format(get_ie_id(tag_param)): str(safe_get(tag_param, "oui", "")),
        "{}_type".format(get_ie_id(tag_param)): convert_hexadecimal(safe_get(tag_param, "type", 0)),
    }


def extract_ie127_value(tag_param):
    """
    Information Element 127: Extended Capabilities.
    """
    extcap = safe_get(tag_param, "extcap", [])
    return {get_ie_id(tag_param): convert_hexadecimal_list(extcap)}


def extract_ie107_value(tag_param):
    """
    Information Element 107: Internetworking Information.
    """
    return {
        "{}_access_network_type".format(get_ie_id(tag_param)): convert_hexadecimal(safe_get(tag_param, "access_network_type", 0)),
        "{}_asra".format(get_ie_id(tag_param)): convert_hexadecimal(safe_get(tag_param, "asra", 0)),
        "{}_internet".format(get_ie_id(tag_param)): convert_hexadecimal(safe_get(tag_param, "internet", 0)),
        "{}_esr".format(get_ie_id(tag_param)): convert_hexadecimal(safe_get(tag_param, "esr", 0)),
        "{}_uesa".format(get_ie_id(tag_param)): convert_hexadecimal(safe_get(tag_param, "uesa", 0)),
    }


def extract_ie191_value(tag_param):
    """
    Information Element 191: VHT Capabilities.
    """
    return {get_ie_id(tag_param): convert_hexadecimal(safe_get(tag_param, "capabilities", 0))}


if __name__ == "__main__":

    # ====================================================================
    #                   PARAMETRI CONVERSIONE DATASET BONN
    # ====================================================================

    input_dir = "/home/giuff/Tesi/TransformerTry/Dataset/Bonn_Dataset/dataset-structure-pseudonymized"
    output_dir = "/home/giuff/Tesi/TransformerTry/Dataset/Bonn_Dataset/json"

    os.makedirs(output_dir, exist_ok=True)

    pcap_files = []

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".pcap"):
                pcap_files.append(os.path.join(root, file))

    print(f"[INFO] Trovati {len(pcap_files)} file PCAP")

    for i, pcap_file in enumerate(pcap_files, start=1):
        pcap_name = os.path.basename(pcap_file)
        json_name = os.path.splitext(pcap_name)[0] + ".json"
        output_json = os.path.join(output_dir, json_name)

        print(f"\n[{i}/{len(pcap_files)}] Converto:")
        print(f"PCAP: {pcap_file}")
        print(f"JSON: {output_json}")

        dataset = extract_from_pcap(pcap_file=pcap_file)

        for record in dataset:
            record["label"] = -1

        with open(output_json, "w") as f:
            json.dump(dataset, f, indent=4)

        print(f"[OK] Salvati {len(dataset)} record")

    print("\nConversione completata.")
