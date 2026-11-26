"""Small RESP2 GET/SET client for the documented internal Cluster Proxy."""
import socket

class RedisIO:
    def __init__(self, host='127.0.0.1', port=31379, timeout=.7):
        self.host=host;self.port=port;self.timeout=timeout;self.connection=None;self.stream=None

    def close(self):
        if self.stream is not None:self.stream.close()
        if self.connection is not None:self.connection.close()
        self.stream=None;self.connection=None

    def command(self, *args):
        try:
            if self.connection is None:
                self.connection=socket.create_connection((self.host,self.port),self.timeout);self.connection.settimeout(self.timeout)
                self.connection.setsockopt(socket.SOL_SOCKET,socket.SO_KEEPALIVE,1);self.stream=self.connection.makefile('rb')
            encoded=[str(arg).encode('utf-8') for arg in args]
            packet=b'*'+str(len(encoded)).encode()+b'\r\n'+b''.join(b'$'+str(len(arg)).encode()+b'\r\n'+arg+b'\r\n' for arg in encoded)
            self.connection.sendall(packet);line=self.stream.readline(256)
            if not line.endswith(b'\r\n'):raise ValueError('Incomplete Redis response')
            kind,value=line[:1],line[1:-2]
            if kind==b'-':raise ValueError('Redis error: '+value.decode(errors='replace')[:180])
            if kind==b'+':return value.decode('utf-8')
            if kind!=b'$':raise ValueError('Unsupported Redis response')
            length=int(value)
            if length==-1:return None
            if not 0<=length<=256*1024:raise ValueError('Redis payload exceeds limit')
            body=self.stream.read(length);ending=self.stream.read(2)
            if len(body)!=length or ending!=b'\r\n':raise ValueError('Incomplete Redis payload')
            return body.decode('utf-8')
        except Exception:
            self.close();raise

    def get(self, key):return self.command('GET',key)

    def set(self, key, value, ttl=15):
        if self.command('SET',key,value,'EX',ttl)!='OK':raise ValueError('Redis SET not acknowledged')
