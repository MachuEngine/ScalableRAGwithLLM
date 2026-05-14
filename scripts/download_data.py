import os
import pandas as pd

def download_nsmc_from_source(save_dir="data/raw"):
    # 저장할 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)
    
    print("GitHub 원본 저장소에서 NSMC 데이터 다운로드 중...")
    train_url = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt"
    test_url = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt"
    
    # pandas를 이용해 원본 TSV(Tab-Separated Values) 파일 직접 읽기
    train_df = pd.read_csv(train_url, sep='\t')
    test_df = pd.read_csv(test_url, sep='\t')
    
    # 결측치(NaN) 제거 (데이터 전처리 기초)
    train_df = train_df.dropna()
    test_df = test_df.dropna()
    
    print(f"다운로드 및 전처리 완료: Train {len(train_df)}건, Test {len(test_df)}건")
    
    # MLOps에서 효율적인 데이터 처리를 위해 Parquet 포맷으로 저장
    train_path = os.path.join(save_dir, "train.parquet")
    test_path = os.path.join(save_dir, "test.parquet")
    
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    print(f"저장 완료: {save_dir} 디렉토리에 parquet 파일이 생성되었습니다.")

if __name__ == "__main__":
    download_nsmc_from_source()