import os

def xor_data(data: bytes, key: int) -> bytes:
    """XORs byte data with a single byte key."""
    return bytes(b ^ key for b in data)

def encrypt_csv_files(directory: str, key: int = 0x5A, encrypted_ext: str = ".enc"):
    """
    Finds all CSV files in the specified directory, encrypts their content
    using XOR, and saves them with a new extension.
    """
    print(f"Starting encryption in directory: {directory}")
    found_csv = False
    for filename in os.listdir(directory):
        if filename.lower().endswith(".csv"):
            found_csv = True
            csv_filepath = os.path.join(directory, filename)
            encrypted_filename = os.path.splitext(filename)[0] + encrypted_ext
            encrypted_filepath = os.path.join(directory, encrypted_filename)

            print(f"Processing: {filename} -> {encrypted_filename}")
            try:
                with open(csv_filepath, 'rb') as f_in: # read bytes
                    original_data = f_in.read()

                encrypted_data = xor_data(original_data, key)

                with open(encrypted_filepath, 'wb') as f_out: # write bytes
                    f_out.write(encrypted_data)
                print(f"Successfully encrypted: {encrypted_filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    if not found_csv:
        print("No CSV files found in the directory.")
    print("Encryption process finished.")

if __name__ == "__main__":
    current_working_directory = os.getcwd()
    # Simple XOR key (example: ASCII 'Z')
    encryption_key = 0x5A
    encrypt_csv_files(current_working_directory, key=encryption_key)
