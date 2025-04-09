import os

def xor_data(data: bytes, key: int) -> bytes:
    """XORs byte data with a single byte key. (Same function decrypts)"""
    return bytes(b ^ key for b in data)

def decrypt_files_to_csv(directory: str, key: int = 0x5A, encrypted_ext: str = ".enc", output_ext: str = ".csv"):
    """
    Finds all encrypted files (with encrypted_ext) in the specified directory,
    decrypts their content using XOR, and saves them with the output_ext (CSV).
    """
    print(f"Starting decryption in directory: {directory}")
    found_encrypted = False
    for filename in os.listdir(directory):
        if filename.lower().endswith(encrypted_ext):
            found_encrypted = True
            encrypted_filepath = os.path.join(directory, filename)
            decrypted_filename = os.path.splitext(filename)[0] + output_ext
            decrypted_filepath = os.path.join(directory, decrypted_filename)

            count = 1
            while os.path.exists(decrypted_filepath):
                base_name = os.path.splitext(filename)[0]
                decrypted_filename = f"{base_name}_decrypted_{count}{output_ext}"
                decrypted_filepath = os.path.join(directory, decrypted_filename)
                count += 1

            print(f"Processing: {filename} -> {decrypted_filename}")
            try:
                with open(encrypted_filepath, 'rb') as f_in: # read bytes
                    encrypted_data = f_in.read()

                decrypted_data = xor_data(encrypted_data, key)

                with open(decrypted_filepath, 'wb') as f_out: # write bytes
                    f_out.write(decrypted_data)
                print(f"Successfully decrypted: {decrypted_filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    if not found_encrypted:
        print(f"No files with extension '{encrypted_ext}' found in the directory.")
    print("Decryption process finished.")

if __name__ == "__main__":
    current_working_directory = os.getcwd()
    # Must be the same key used for encryption
    decryption_key = 0x5A
    encrypted_file_extension = ".enc"
    decrypt_files_to_csv(
        current_working_directory,
        key=decryption_key,
        encrypted_ext=encrypted_file_extension
    )
