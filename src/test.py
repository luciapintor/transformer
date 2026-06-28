if __name__ == "__main__":
    import json
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract Wi-Fi Probe Request features from a PCAP file and save them to JSON."
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path del file PCAP di input"
    )

    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path del file JSON di output"
    )

    args = parser.parse_args()

    packet_list_summary = extract_from_pcap(pcap_file=args.input)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(packet_list_summary, f, indent=4, ensure_ascii=False)

    print(f"Extracted {len(packet_list_summary)} packets")
    print(f"Saved output to: {args.output}")