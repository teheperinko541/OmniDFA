#!/bin/bash
# Unpack script with flexible parameters:
#   Usage: ./unpack_all.sh <target_dir> [source_dir]
#   - target_dir:  Where to restore the train/ and val/ folders (required).
#   - source_dir:  Directory containing the .zip and .z01... files (optional, defaults to current directory).

TARGET_DIR="$1"
SOURCE_DIR="${2:-.}"   # Default to current directory if not given

# Validate target directory
if [ -z "$TARGET_DIR" ]; then
    echo "Error: Target directory is required."
    echo "Usage: $0 <target_dir> [source_dir]"
    exit 1
fi

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory '$SOURCE_DIR' does not exist."
    exit 1
fi

# Create target directory (and parent directories if needed)
mkdir -p "$TARGET_DIR"

# Move into source directory
cd "$SOURCE_DIR" || exit 1

echo "Source directory (containing .zip files): $SOURCE_DIR"
echo "Target extraction directory: $TARGET_DIR"
echo "----------------------------------------------"

# Loop over all .zip files (main archive files)
for zip_file in *.zip; do
    # Skip if no .zip files exist
    [ -e "$zip_file" ] || { echo "No .zip files found in '$SOURCE_DIR'."; break; }

    echo "Extracting $zip_file ..."
    unzip -o "$zip_file" -d "$TARGET_DIR"
    if [ $? -eq 0 ]; then
        echo "  -> $zip_file extracted successfully."
    else
        echo "  -> Warning: $zip_file extraction encountered issues."
    fi
done

echo "----------------------------------------------"
echo "All extractions completed. Restored data is in: $TARGET_DIR"
echo "Contents should include train/ and val/ folders (if all parts were present)."