import pytest
from src.extensions.ext_cryptographer import ext_Cryptographer

class TestCryptographer:

    @pytest.fixture
    def crypto(self):
        """Fresh cryptographer for each test"""
        key = b"test_key_minimum_16_bytes_long"
        return ext_Cryptographer(key)

    def test_encryption_preserves_data(self, crypto):
        """Verify round-trip encryption/decryption"""
        # Test with Gene-sized data (96 bytes)
        data = b"A" * 96

        encrypted = crypto.encrypt(data)
        assert encrypted != data  # Must be different

        # Create fresh instance to simulate storage round-trip
        crypto2 = ext_Cryptographer(crypto.user_key)
        decrypted = crypto2.decrypt(encrypted)

        assert decrypted == data

    def test_gyration_evolution_affects_encryption(self, crypto):
        """Verify gyration evolution changes ciphertext"""
        data = b"test data"

        # First encryption
        enc1 = crypto.encrypt(data)

        # Evolve gyration through navigation events (do not reset counter)
        for i in range(8):
            crypto.process_navigation_event(i * 17)  # Simulate navigation

        # Second encryption with evolved gyration
        enc2 = crypto.encrypt(data)

        assert enc1 != enc2  # Different due to gyration change

    def test_16_byte_gyration_evolution(self, crypto):
        """Verify gyration evolves every 16 bytes during encryption"""
        # Track gyration changes
        gyro_cryptography_values = [crypto.gyro_cryptography]

        # Encrypt 48 bytes (should trigger 2 gyration evolutions)
        data = b"X" * 48
        crypto.encrypt(data)

        # Check evolution count
        stats = crypto.ext_get_crypto_analysis()
        # Initial gyro_cryptography + 2 evolutions (at 16 and 32 bytes)
        assert stats["statistics"]["evolution_count"] >= 2

    def test_decrypt_is_stateless(self, crypto):
        """Verify decrypt doesn't affect future encryptions"""
        data = b"test"

        # Encrypt
        initial_counter = crypto.counter
        encrypted = crypto.encrypt(data)
        counter_after_encrypt = crypto.counter

        # Decrypt
        decrypted = crypto.decrypt(encrypted)
        counter_after_decrypt = crypto.counter

        # Counter should advance after encrypt but not after decrypt
        assert counter_after_encrypt > initial_counter
        assert counter_after_decrypt == counter_after_encrypt
        assert decrypted == data

    def test_footprint_is_5_bytes(self, crypto):
        """Verify memory footprint"""
        assert crypto.get_footprint_bytes() == 5

    def test_navigation_integration(self, crypto):
        """Verify navigation events affect crypto state"""
        initial_gyro_cryptography = crypto.gyro_cryptography

        # Process 8 navigation events (triggers gyro_cryptography evolution)
        for i in range(8):
            crypto.process_navigation_event(0x42)

        assert crypto.gyro_cryptography != initial_gyro_cryptography
