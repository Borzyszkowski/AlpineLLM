#!/usr/bin/env bash
#
# A script to download text datasets for AlpineLLM experiments.
#

set -e

print_usage() {
    echo "Usage: $0 [dataset_type]"
    echo
    echo "dataset_type:"
    echo "  alpine (default)  - Downloads the alpine dataset"
    echo "  shakespeare       - Downloads the tiny Shakespeare dataset"
    echo
}

# If -h or --help was passed, show usage
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    print_usage
    exit 0
fi

# Dataset argument (default = alpine)
DATASET="${1:-alpine}"

# URLs per dataset
ALPINE_URLS=(
    "https://www.gutenberg.org/cache/epub/69128/pg69128.txt" # The making of a mountaineer
    "https://www.gutenberg.org/cache/epub/47209/pg47209.txt" # Hours of exercise in the Alps
)
SHAKESPEARE_URLS=(
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)

# Map dataset to URL list and default dir
case "$DATASET" in
    alpine)
        URLS=("${ALPINE_URLS[@]}")
        TARGET_DIR="./raw_data/alpine"
        ;;
    shakespeare)
        URLS=("${SHAKESPEARE_URLS[@]}")
        TARGET_DIR="./raw_data/shakespeare"
        ;;
    *)
        echo "Unknown dataset type: $DATASET"
        print_usage
        exit 1
        ;;
esac

# Create directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Download datasets
for URL in "${URLS[@]}"; do
    FILENAME=$(basename "$URL")
    wget -O "$TARGET_DIR/$FILENAME" "$URL"
    echo "Downloaded $FILENAME to $TARGET_DIR"
done

echo "All $DATASET dataset files saved to $TARGET_DIR"
