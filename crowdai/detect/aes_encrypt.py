import base64
import hashlib
from Crypto.Cipher import AES
from Crypto import Random

class AESCipher(object):
    def __init__(self , IV = '0123456789abcdef'):
        self.bs = 16
        self.key = '0123456789abcdef'
        self.cipher = AES.new(self.key.encode('utf-8'), AES.MODE_ECB)

    def encrypt(self, raw):
        raw = self._pad(raw)
        encrypted = self.cipher.encrypt(str.encode(raw))
        encoded = base64.b64encode(encrypted)
        return str(encoded, 'utf-8')

    def decrypt(self, raw):
        decoded = base64.b64decode(raw)
        decrypted = self.cipher.decrypt(decoded)
        return str(self._unpad(decrypted), 'utf-8')

    def _pad(self, s):
        return s + (self.bs - len(s) % self.bs) * chr(self.bs - len(s) % self.bs)

    def _unpad(self, s):
        return s[:-ord(s[len(s)-1:])]



#
# import secrets
# authtoken = ''.join([secrets.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789#@!$%*') for i in range(8)])
# print(authtoken)
#
# key = 'yyyyyyyyyyyyyyyy'
# key = '0123456789abcdef'
# IV= b'0123456789abcdef'   #need to be encoded too
# cipher = AESCipher(key)
# plaintext = "TECHM00000001&8130141308&{}".format(authtoken)
# print(plaintext)
# encrypted = cipher.encrypt(plaintext)
# print(encrypted)
# print('Encrypted: %s' % encrypted)

# decrypted = cipher.decrypt(encrypted)
# print('Decrypted: %s' % decrypted)