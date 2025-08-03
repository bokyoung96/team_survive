### Protocols (인터페이스 계층)

```
DataFetcher (Protocol)
├── fetch(symbol: Symbol, timeframe: TimeFrame, time_range: Optional[TimeRange]) -> pd.DataFrame
└── 목적: 데이터 수집 인터페이스 정의

Repository (Protocol)
├── load(symbol: Symbol, timeframe: TimeFrame, data_type: DataType) -> Optional[pd.DataFrame]
├── save(data: pd.DataFrame, symbol: Symbol, timeframe: TimeFrame, data_type: DataType) -> None
└── 목적: 데이터 저장/로드 인터페이스 정의
```

### Models (데이터 구조 계층)

```
DataType (Enum)
├── OHLCV = "ohlcv"
└── 목적: 지원하는 데이터 타입 정의

TimeFrame (Enum)
├── M1 = "1m", M5 = "5m", M15 = "15m", M30 = "30m"
├── H1 = "1h", H4 = "4h", D1 = "1d", W1 = "1w"
└── 목적: 시간 프레임 정의

MarketType (Enum)
├── SPOT = "spot", SWAP = "swap", FUTURE = "future"
└── 목적: 거래소 마켓 타입 정의

Symbol (frozen dataclass)
├── base: str                    # 기본 화폐 (BTC)
├── quote: str                   # 견적 화폐 (USDT)
├── settle: Optional[str]        # 정산 화폐 (USDT)
├── to_string(market_type) -> str # 거래소 형식으로 변환
└── 목적: 거래 쌍 정보 표현

TimeRange (frozen dataclass)
├── start: datetime              # 시작 시간
├── end: datetime                # 종료 시간
├── __post_init__()              # 시간대 처리 및 검증
├── days(n, end) -> TimeRange    # n일 전부터 end까지
└── hours(n, end) -> TimeRange   # n시간 전부터 end까지

Exchange (dataclass)
├── id: str                      # 거래소 ID (binance)
├── name: Optional[str]          # 거래소 이름
├── default_type: MarketType     # 기본 마켓 타입
├── enable_rate_limit: bool      # 요청 제한 활성화
├── options: Dict[str, Any]      # 추가 옵션
├── config -> Dict[str, Any]     # CCXT 설정
└── client() -> ccxt.Exchange    # CCXT 클라이언트 생성
```

### Implementations (구현체 계층)

```
OHLCVFetcher (DataFetcher 구현)
├── _exchange: Exchange          # 거래소 정보
├── _market_type: MarketType     # 마켓 타입
├── _client: ccxt.Exchange       # CCXT 클라이언트
├── fetch() -> pd.DataFrame      # 데이터 수집 메인 메서드
├── _fetch_once() -> pd.DataFrame # 단일 요청
├── _fetch_multi() -> pd.DataFrame # 페이지네이션 요청
└── _to_df() -> pd.DataFrame     # 데이터프레임 변환

RegistryOrchestrator (Repository 구현)
├── _base_path: Path             # 저장 기본 경로
├── _exchange: Exchange          # 거래소 정보
├── load() -> Optional[pd.DataFrame] # 데이터 로드
├── save() -> None               # 데이터 저장
└── _get_path() -> Path          # 파일 경로 생성
```

### Factory & Loader (조합 계층)

```
Factory
├── _exchange: Exchange          # 거래소 정보
├── _base_path: Path             # 저장 경로
├── create_fetcher() -> DataFetcher # Fetcher 생성
└── create_repository() -> Repository # Repository 생성

DataLoader
├── _factory: Factory            # 객체 생성 팩토리
├── _repository: Repository      # 데이터 저장소
├── load() -> pd.DataFrame       # 데이터 로딩 메인 메서드
└── _is_data_sufficient() -> bool # 데이터 충분성 검사
```

## 데이터 흐름

### 1. 데이터 로딩 프로세스

```
사용자 요청
    ↓
DataLoader.load()
    ↓
Repository.load() (로컬 확인)
    ↓
데이터 있음? ──YES──> 반환
    ↓ NO
Factory.create_fetcher()
    ↓
Fetcher.fetch() (외부 API 호출)
    ↓
Repository.save() (로컬 저장)
    ↓
데이터 반환
```

### 2. 페이지네이션 로직

```
time_range 지정
    ↓
_fetch_multi() 호출
    ↓
start_ts부터 end_ts까지 반복
    ↓
fetch_ohlcv(since=since) 호출
    ↓
데이터 수집 및 last_ts 업데이트
    ↓
end_ts 도달 또는 데이터 없음까지 반복
```

## 확장 방법

### 새로운 데이터 타입 추가

1. **models.py**: `DataType` Enum에 추가

   ```python
   @unique
   class DataType(Enum):
       OHLCV = "ohlcv"
       FUNDING_RATE = "funding_rate"  # 새로 추가
   ```

2. **fetchers.py**: 새로운 Fetcher 클래스 구현

   ```python
   class FundingRateFetcher(DataFetcher):
       def fetch(self, symbol, timeframe, time_range=None):
           # 구현 로직
   ```

3. **factory.py**: `create_fetcher`에 분기 추가
   ```python
   def create_fetcher(self, data_type: DataType) -> DataFetcher:
       if data_type == DataType.OHLCV:
           return OHLCVFetcher(self._exchange)
       elif data_type == DataType.FUNDING_RATE:  # 새로 추가
           return FundingRateFetcher(self._exchange)
   ```

### 새로운 저장소 타입 추가

1. **Repository Protocol 구현**

   ```python
   class DatabaseRepository(Repository):
       def load(self, symbol, timeframe, data_type):
           # DB에서 로드
       def save(self, data, symbol, timeframe, data_type):
           # DB에 저장
   ```

2. **Factory 수정**
   ```python
   def create_repository(self) -> Repository:
       return DatabaseRepository(self._connection)
   ```

## 주요 특징

### 1. 불변성 (Immutability)

- `frozen=True` dataclass 사용으로 객체 상태 변경 방지
- 예측 가능한 동작과 스레드 안전성 보장

### 2. 지연 초기화 (Lazy Initialization)

- `Exchange.client()`: 필요할 때만 CCXT 클라이언트 생성
- 메모리 효율성과 초기화 시간 단축

### 3. 자동 파일 경로 생성

- `RegistryOrchestrator._get_path()`: 데이터 타입, 심볼, 타임프레임 기반 자동 경로 생성
- 일관된 파일 구조 유지

### 4. 에러 처리

- `TimeRange.__post_init__()`: 시간 범위 유효성 검증
- `parse_symbol()`: 심볼 형식 검증

## 사용 예시

```python
# 기본 설정
exchange = Exchange(id="binance", default_type=MarketType.SWAP)
factory = Factory(exchange=exchange, base_path=Path("./data"))
loader = DataLoader(factory=factory)

# 심볼 및 시간 범위 설정
symbol = parse_symbol("BTC/USDT:USDT")
time_range = TimeRange.days(n=7)

# 데이터 로딩
ohlcv_data = loader.load(
    symbol=symbol,
    timeframe=TimeFrame.M5,
    data_type=DataType.OHLCV,
    time_range=time_range
)
```
