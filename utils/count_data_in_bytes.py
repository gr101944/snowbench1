def count_bytes_in_string(input_string):
    try:
        byte_count = len(input_string.encode('utf-8'))
        print(f'The number of bytes in the string is: {byte_count}')
        return byte_count 
        
    except Exception as e:
        print(f"An error occurred: {e}")