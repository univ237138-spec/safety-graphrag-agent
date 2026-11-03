"""KOSHA 사망사고 고위험요인(SIF) 아카이브 수집 → data/raw/sif.json

API는 재해개요 검색어가 필수라 넓은 검색어 여러 개로 받아 합치고 중복을 뺀다.
키: .env의 DATA_GO_KR_KEY (인코딩·디코딩 키 모두 가능). 실행: python collect.py
"""
import json
import pathlib
import urllib.parse
import urllib.request

URL = "https://apis.data.go.kr/B552468/sifArchive/view"
KEYWORDS = ["년", "재해자", "사망", "중", "작업", "에서", "하", "다", "OO", "경"]
FIELDS = ["cateSeNm", "sifLclsfNm", "sifMclsfNm", "sifSclsfNm", "disasterType",
          "disasterOverview", "orgtNm", "situation", "disasterFactor", "dcrsCntrplnCn"]
ROOT = pathlib.Path(__file__).parent
OUT = ROOT / "data" / "raw" / "sif.json"


def load_key():
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("DATA_GO_KR_KEY="):
            return urllib.parse.unquote(line.split("=", 1)[1].strip())
    raise SystemExit(".env에 DATA_GO_KR_KEY가 없습니다.")


def fetch(key, keyword, page, rows=1000):
    query = urllib.parse.urlencode({"serviceKey": key, "callApiId": 1170, "cateSeCd": 1,
                                    "disasterOverview": keyword, "pageNo": page, "numOfRows": rows})
    with urllib.request.urlopen(f"{URL}?{query}", timeout=60) as response:
        data = json.loads(response.read())
    if data["header"]["resultCode"] != "00":
        raise RuntimeError(f"{keyword} p{page}: {data['header']}")
    body = data["body"]
    items = body["items"]["item"] if body.get("items") else []
    return int(body["totalCount"]), items if isinstance(items, list) else [items]


def main():
    key = load_key()
    records = {}
    for keyword in KEYWORDS:
        page, total = 1, None
        while total is None or (page - 1) * 1000 < total:
            total, items = fetch(key, keyword, page)
            for item in items:
                row = {f: item.get(f) for f in FIELDS}
                records.setdefault(tuple(row.values()), row)
            page += 1
        print(f"{keyword}: {total}건 → 누적 {len(records)}건")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(list(records.values()), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장: {OUT} ({len(records)}건)")


if __name__ == "__main__":
    main()
