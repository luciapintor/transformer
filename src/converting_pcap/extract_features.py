# script by lucia pintor
import os

import pyshark as ps
from datetime import datetime

from convert_hexadecimal import convert_hexadecimal, convert_hexadecimal_list

def extract_from_pcap(pcap_file):
    """
    This function extracts the features from the pcap file and returns a list of packets with their features.
    The features are:
    - timestamp
    - mac address
    - Information Elements (IEs) of the packet
    The IEs are extracted from the WLAN management layer of the packet.
    """
    
    packet_list_summary = []
    
    try:
        packet_list = ps.FileCapture(input_file=pcap_file, use_json=True)
    
        for pkt in packet_list:
            pkt_summary = {}
            pkt_summary['timestamp'] = extract_timestamp(pcap_frame=pkt)
            pkt_summary['mac'] = extract_mac(pcap_frame=pkt)
            pkt_summary['seq'] = int(pkt.wlan.seq)
            pkt_summary.update(extract_tag_paramenters(pcap_frame=pkt))
            packet_list_summary.append(pkt_summary)
            
        # close the capture file
        packet_list.close()
        
    except Exception as error:
        raise error
    
    return packet_list_summary

def extract_timestamp(pcap_frame):
    timestamp_str = pcap_frame.sniff_timestamp
    timestamp_str = timestamp_str[:26]
    timestamp_datetime = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%f")
    return timestamp_datetime.timestamp()

def extract_mac(pcap_frame):
    return pcap_frame.wlan.sa

def extract_wlan_mgt_layer(pcap_frame):
    return pcap_frame.layers[3]

def extract_tag_paramenters(pcap_frame):
    """
    This function extracts the Information Elements (IEs) from the WLAN management layer of the packet.
    """
    ie_dict = {} # tags are the Information Elements
    
    tag_list = extract_wlan_mgt_layer(pcap_frame).all.tag
    
    index_221 = 0 
    index_127 = 0 
    
    for t in tag_list:
        ie_info = extract_ie(t)
        ie_info_keys = list(ie_info.keys())
        
        # IEs 221 can have multiple instances in the same packet
        if "221_oui" in ie_info_keys[0]: # if there is a 221 tag, it means that there is a oui and a type
            for ie_key in ie_info_keys:
                ie_dict["{}_{}".format(ie_key, index_221)] = ie_info[ie_key]
            index_221 += 1

        elif "127" in ie_info_keys[0]:
            for ie_key in ie_info_keys:
                ie_dict["{}_{}".format(ie_key, index_127)] = ie_info[ie_key]
                index_127 += 1
            
        else:
            if ie_info_keys[0] in ie_dict:
                print("Warning: tag {} already exists in the dictionary".format(ie_info_keys[-1]))
            ie_dict.update(ie_info)
        
    return ie_dict

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
        return {get_ie_id(tag_param): 0}

def get_ie_id(tag_param):
    return "ie{}".format(tag_param.number)
    
def extract_ie0_value(tag_param):
    """
    This function extracts the Information Element 0, which contains the SSID of the packet.
    """
    return {get_ie_id(tag_param): tag_param.ssid}

def extract_ie1_value(tag_param):
    """
    This function extracts the Information Element 1, which contains the supported rates of the packet.
    """
    return {get_ie_id(tag_param): convert_hexadecimal_list(tag_param.supported_rates)}

def extract_ie3_value(tag_param):
    """
    This function extracts the Information Element 3, which contains the current channel of the packet.
    """
    return {get_ie_id(tag_param): tag_param.current_channel}

def extract_ie45_value(tag_param):
    """
    This function extracts the Information Element 45, which contains numerous fields about the Antenna Capabilities.
    """
    
    # convert params of type tuple
    ampduparam = convert_hexadecimal(''.join(tag_param.ampduparam))
    asel = convert_hexadecimal(''.join(tag_param.asel))
    capabilities = convert_hexadecimal(''.join(tag_param.capabilities))
    txbf = convert_hexadecimal(''.join(tag_param.txbf))
    
    # mcsset fields
    mcsset_txunequalmod = int(tag_param.mcsset.txunequalmod)
    mcsset_txrxmcsnotequal = int(tag_param.mcsset.txrxmcsnotequal)
    mcsset_txmaxss = convert_hexadecimal(tag_param.mcsset.txmaxss)
    mcsset_txsetdefined = int(tag_param.mcsset.txsetdefined)
    mcsset_highestdatarate = convert_hexadecimal(tag_param.mcsset.highestdatarate)
    
    # mcsset_rxbitmask fields
    rxbitmask_0to7 = convert_hexadecimal(tag_param.mcsset.rxbitmask.get('0to7', 0))
    rxbitmask_8to15 = convert_hexadecimal(tag_param.mcsset.rxbitmask.get('8to15', 0))
    rxbitmask_16to23 = convert_hexadecimal(tag_param.mcsset.rxbitmask.get('16to23', 0))
    rxbitmask_24to31 = convert_hexadecimal(tag_param.mcsset.rxbitmask.get('24to31', 0))
    rxbitmask_32 = convert_hexadecimal(tag_param.mcsset.rxbitmask.get('32', 0))
    rxbitmask_33to38 = convert_hexadecimal(tag_param.mcsset.rxbitmask.get('33to38', 0))
    rxbitmask_39to52 = convert_hexadecimal(tag_param.mcsset.rxbitmask.get('39to52', 0))
    rxbitmask_53to76 = convert_hexadecimal(tag_param.mcsset.rxbitmask.get('53to76', 0))
    
    return {
        '{}_ampduparam'.format(get_ie_id(tag_param)): ampduparam,
        '{}_asel'.format(get_ie_id(tag_param)): asel,
        '{}_capabilities'.format(get_ie_id(tag_param)): capabilities,
        '{}_txbf'.format(get_ie_id(tag_param)): txbf,
        
        '{}_mcsset_txunequalmod'.format(get_ie_id(tag_param)): mcsset_txunequalmod,
        '{}_mcsset_txrxmcsnotequal'.format(get_ie_id(tag_param)): mcsset_txrxmcsnotequal,
        '{}_mcsset_txmaxss'.format(get_ie_id(tag_param)): mcsset_txmaxss,
        '{}_mcsset_txsetdefined'.format(get_ie_id(tag_param)): mcsset_txsetdefined,
        '{}_mcsset_highestdatarate'.format(get_ie_id(tag_param)): mcsset_highestdatarate,
        
        '{}_rxbitmask_0to7'.format(get_ie_id(tag_param)): rxbitmask_0to7,
        '{}_rxbitmask_8to15'.format(get_ie_id(tag_param)): rxbitmask_8to15,
        '{}_rxbitmask_16to23'.format(get_ie_id(tag_param)): rxbitmask_16to23,
        '{}_rxbitmask_24to31'.format(get_ie_id(tag_param)): rxbitmask_24to31,
        '{}_rxbitmask_32'.format(get_ie_id(tag_param)): rxbitmask_32,
        '{}_rxbitmask_33to38'.format(get_ie_id(tag_param)): rxbitmask_33to38,
        '{}_rxbitmask_39to52'.format(get_ie_id(tag_param)): rxbitmask_39to52,
        '{}_rxbitmask_53to76'.format(get_ie_id(tag_param)): rxbitmask_53to76,
        
        }

def extract_ie50_value(tag_param):
    """
    This function extracts the Information Element 50, which contains the extended supported rates of the packet.
    """
    return {get_ie_id(tag_param): convert_hexadecimal_list(tag_param.extended_supported_rates)}

def extract_ie221_value(tag_param):
    """
    This function extracts the Information Element 221, which contains the oui (Orgazination Unique ID) and the vendor specific information.
    """
    return {
        '{}_oui'.format(get_ie_id(tag_param)): tag_param.oui, 
        '{}_type'.format(get_ie_id(tag_param)): tag_param.type,
        }

def extract_ie127_value(tag_param):
    """
    This function extracts the Information Element 127, which contains the HT Capabilities of the packet.
    """
    extcap = convert_hexadecimal(tag_param.extcap)
    
    if isinstance(extcap, list):
        return {
            '{}'.format(get_ie_id(tag_param)): convert_hexadecimal_list(extcap),
            }
    if isinstance(extcap, int):
        return {
            '{}'.format(get_ie_id(tag_param)): [extcap],
            }
    else:
        return {
            '{}'.format(get_ie_id(tag_param)): convert_hexadecimal(extcap),
            }

def extract_ie107_value(tag_param):
    """
    This function extracts the Information Element 107, which contains the Internetworking Information of the packet.
    """
    access_network_type = tag_param.access_network_type
    internet = tag_param.internet   
    asra = tag_param.asra
    esr = tag_param.esr
    uesa = tag_param.uesa
    # hessid = tag_param.hessid  

    
    return {
        '{}_access_network_type'.format(get_ie_id(tag_param)): access_network_type, 
        '{}_asra'.format(get_ie_id(tag_param)): asra,
        '{}_internet'.format(get_ie_id(tag_param)): internet,    
        '{}_esr'.format(get_ie_id(tag_param)): esr,
        '{}_uesa'.format(get_ie_id(tag_param)): uesa,
        # '{}_hessid'.format(get_ie_id(tag_param)): hessid,
    }
    
def extract_ie191_value(tag_param):
    """
    This function extracts the Information Element 191, which contains the VHT (Very High Throughput) Capabilities of the packet.
    """
    return {
        '{}'.format(get_ie_id(tag_param)): convert_hexadecimal(tag_param.capabilities),
        }
    
    
if __name__ == "__main__":
    packet_list_summary = extract_from_pcap(pcap_file="src/converting_pcap/example.pcap")
    
    print("Printing the first 5 packets with their features:")
    
    for i, packet in enumerate(packet_list_summary[:5]):
        print(f"Packet {i+1}: ")
        for key, value in packet.items():
            print(f"  {key}: {value}")