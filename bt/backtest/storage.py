import json
import numpy as np
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Trade:
    timestamp: int
    symbol: str
    side: str
    price: float
    quantity: float
    pnl: float
    cumulative_pnl: float
    position: float
    trade_id: int
    
    def to_bytes(self) -> bytes:
        # NOTE: Fixed size: 8+32+4+8+8+8+8+8+4 = 88 bytes
        data = np.zeros(1, dtype=[
            ('timestamp', 'i8'),
            ('symbol', 'S32'),
            ('side', 'S4'),
            ('price', 'f8'),
            ('quantity', 'f8'),
            ('pnl', 'f8'),
            ('cumulative_pnl', 'f8'),
            ('position', 'f8'),
            ('trade_id', 'i4')
        ])
        
        data['timestamp'] = self.timestamp
        data['symbol'] = self.symbol.encode('utf-8')[:32]
        data['side'] = self.side.encode('utf-8')[:4]
        data['price'] = self.price
        data['quantity'] = self.quantity
        data['pnl'] = self.pnl
        data['cumulative_pnl'] = self.cumulative_pnl
        data['position'] = self.position
        data['trade_id'] = self.trade_id
        
        return data.tobytes()
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'Trade':
        arr = np.frombuffer(data, dtype=[
            ('timestamp', 'i8'),
            ('symbol', 'S32'),
            ('side', 'S4'),
            ('price', 'f8'),
            ('quantity', 'f8'),
            ('pnl', 'f8'),
            ('cumulative_pnl', 'f8'),
            ('position', 'f8'),
            ('trade_id', 'i4')
        ])
        
        return cls(
            timestamp=int(arr['timestamp'][0]),
            symbol=arr['symbol'][0].decode('utf-8').rstrip('\x00'),
            side=arr['side'][0].decode('utf-8').rstrip('\x00'),
            price=float(arr['price'][0]),
            quantity=float(arr['quantity'][0]),
            pnl=float(arr['pnl'][0]),
            cumulative_pnl=float(arr['cumulative_pnl'][0]),
            position=float(arr['position'][0]),
            trade_id=int(arr['trade_id'][0])
        )


class TradeStorage:
    TRADE_SIZE = 88
    
    def __init__(self, base_dir: str = "bt_results"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        self.mmap_file = None
        self.mmap_array = None
        self.mmap_path = None
        self.trade_count = 0
        self.session_id = None
    
    def initialize_session(self, session_id: str):
        self.session_id = session_id
        
        session_dir = self.base_dir / session_id
        session_dir.mkdir(exist_ok=True)
        
        self.mmap_path = session_dir / "trades.dat"
        
        initial_capacity = 10000
        file_size = initial_capacity * self.TRADE_SIZE
        
        if self.mmap_path.exists():
            self.mmap_path.unlink()
        
        with open(self.mmap_path, 'wb') as f:
            f.write(b'\x00' * file_size)
        
        self.mmap_file = open(self.mmap_path, 'r+b')
        self.mmap_array = np.memmap(
            self.mmap_file,
            dtype='uint8',
            mode='r+',
            shape=(initial_capacity, self.TRADE_SIZE)
        )
        
        self.trade_count = 0
        
        metadata = {
            'session_id': session_id,
            'created_at': datetime.now().isoformat(),
            'trade_size': self.TRADE_SIZE
        }
        
        with open(session_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def record_trade(self, trade: Trade):
        if self.mmap_array is None:
            raise RuntimeError("Session not initialized. Call initialize_session first.")
        
        if self.trade_count >= len(self.mmap_array):
            self._expand_mmap()
        
        trade_bytes = trade.to_bytes()
        self.mmap_array[self.trade_count] = np.frombuffer(trade_bytes, dtype='uint8')
        self.trade_count += 1
        
        if self.trade_count % 100 == 0:
            self.mmap_array.flush()
    
    def _expand_mmap(self):
        current_capacity = len(self.mmap_array)
        new_capacity = current_capacity * 2
        
        self.mmap_array.flush()
        del self.mmap_array
        self.mmap_file.close()
        
        with open(self.mmap_path, 'r+b') as f:
            f.seek(0, 2)
            f.write(b'\x00' * (current_capacity * self.TRADE_SIZE))
        
        self.mmap_file = open(self.mmap_path, 'r+b')
        self.mmap_array = np.memmap(
            self.mmap_file,
            dtype='uint8',
            mode='r+',
            shape=(new_capacity, self.TRADE_SIZE)
        )
    
    def load_trades(self, session_id: Optional[str] = None) -> List[Trade]:
        if session_id is None:
            session_id = self.session_id
        
        if session_id is None:
            raise ValueError("No session_id provided")
        
        session_dir = self.base_dir / session_id
        trades_path = session_dir / "trades.dat"
        
        if not trades_path.exists():
            return []
        
        trades = []
        with open(trades_path, 'rb') as f:
            while True:
                data = f.read(self.TRADE_SIZE)
                if not data or data == b'\x00' * self.TRADE_SIZE:
                    break
                try:
                    trade = Trade.from_bytes(data)
                    trades.append(trade)
                except:
                    break
        
        return trades
    
    def close(self):
        if self.mmap_array is not None:
            self.mmap_array.flush()
            del self.mmap_array
        
        if self.mmap_file is not None:
            self.mmap_file.close()
        
        if self.session_id:
            session_dir = self.base_dir / self.session_id
            info_path = session_dir / "info.json"
            
            info = {
                'total_trades': self.trade_count,
                'closed_at': datetime.now().isoformat()
            }
            
            with open(info_path, 'w') as f:
                json.dump(info, f, indent=2)
        
        self.mmap_array = None
        self.mmap_file = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()