import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, concatenate, Dropout, LSTM # LSTM은 선택적
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# --- 1. 하이퍼파라미터 및 설정 ---
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
SEQUENCE_LENGTH = 5 # 강수량 CSV에서 사용할 과거 시간 스텝 수

# 생성된 데이터가 있는 폴더 경로로 수정해야 합니다.
# 예시: BASE_OUTPUT_DIR = 'data_large' (데이터 생성 스크립트에서 사용한 폴더명)
DATA_DIR = 'data/' # 생성된 데이터가 있는 기본 디렉토리
RAINFALL_DIR = os.path.join(DATA_DIR, 'rainfall_data/')
IMAGE_DIR = os.path.join(DATA_DIR, 'image_data/')
LABEL_FILE = os.path.join(DATA_DIR, 'labels.csv')

EPOCHS = 50 # 데이터가 많아졌으므로 에포크 수 증가 가능
BATCH_SIZE = 16 # 배치 크기도 증가 가능
LEARNING_RATE = 0.001

# --- 2. 데이터 로딩 및 전처리 함수 ---
def load_rainfall_data_from_file(filepath, sequence_length):
    try:
        df = pd.read_csv(filepath)
        rainfall_values = df['rainfall_mm'].values
        if len(rainfall_values) >= sequence_length:
            rainfall_sequence = rainfall_values[-sequence_length:]
        else:
            padding = np.zeros(sequence_length - len(rainfall_values))
            rainfall_sequence = np.concatenate([padding, rainfall_values])
        return rainfall_sequence.astype('float32')
    except Exception as e:
        print(f"Error loading rainfall data {filepath}: {e}")
        return np.zeros(sequence_length).astype('float32')

def load_image_data_from_file(filepath, img_width, img_height):
    try:
        img = Image.open(filepath).convert('RGB' if IMG_CHANNELS == 3 else 'L')
        img = img.resize((img_width, img_height))
        img_array = np.array(img) / 255.0
        return img_array.astype('float32')
    except Exception as e:
        print(f"Error loading image {filepath}: {e}")
        return np.zeros((img_height, img_width, IMG_CHANNELS if IMG_CHANNELS == 3 else 1)).astype('float32')

# --- 3. 데이터셋 준비 ---
print("데이터셋 준비 중...")
try:
    labels_df = pd.read_csv(LABEL_FILE)
except FileNotFoundError:
    print(f"레이블 파일({LABEL_FILE})을 찾을 수 없습니다. 경로를 확인해주세요.")
    exit()
except pd.errors.EmptyDataError:
    print(f"레이블 파일({LABEL_FILE})이 비어 있거나 파싱할 데이터가 없습니다.")
    exit()

all_rainfall_sequences = []
all_image_arrays = []
all_target_labels = []

# labels_df의 처음 N개만 사용하려면 .head(N) 사용 (테스트용)
# labels_df = labels_df.head(500) # 예: 처음 500개 샘플만 로드

for index, row in labels_df.iterrows():
    # datetime 컬럼이 있는지 확인 (데이터 생성 스크립트에는 포함되어 있음)
    current_datetime_str = row.get('datetime', 'N/A') # 없으면 'N/A'

    rainfall_filename = row['rainfall_filename']
    image_filename = row['image_filename']
    label = row['is_flooded']

    rainfall_filepath = os.path.join(RAINFALL_DIR, rainfall_filename)
    image_filepath = os.path.join(IMAGE_DIR, image_filename)

    if not os.path.exists(rainfall_filepath):
        print(f"Warning: Rainfall file not found: {rainfall_filepath}. Skipping for datetime: {current_datetime_str}")
        continue
    if not os.path.exists(image_filepath):
        print(f"Warning: Image file not found: {image_filepath}. Skipping for datetime: {current_datetime_str}")
        continue

    rainfall_sequence = load_rainfall_data_from_file(rainfall_filepath, SEQUENCE_LENGTH)
    image_array = load_image_data_from_file(image_filepath, IMG_WIDTH, IMG_HEIGHT)

    all_rainfall_sequences.append(rainfall_sequence)
    all_image_arrays.append(image_array)
    all_target_labels.append(label)

    if (index + 1) % 500 == 0: # 진행 상황 표시
        print(f"데이터 로딩 진행: {index + 1} / {len(labels_df)}")


if not all_target_labels:
    print("처리할 데이터를 로드하지 못했습니다. 데이터 파일 및 경로를 확인해주세요.")
    exit()

X_rainfall = np.array(all_rainfall_sequences)
X_image = np.array(all_image_arrays)
y = np.array(all_target_labels)


if X_rainfall.shape[0] > 0:
    scaler = StandardScaler()
    original_shape = X_rainfall.shape
    X_rainfall_reshaped = X_rainfall.reshape(-1, original_shape[-1])
    X_rainfall_scaled_reshaped = scaler.fit_transform(X_rainfall_reshaped)
    X_rainfall_scaled = X_rainfall_scaled_reshaped.reshape(original_shape)
else:
    X_rainfall_scaled = X_rainfall


# --- 데이터셋 분할 (훈련, 검증, 테스트) ---
X_rain_train, X_img_train, y_train = np.array([]), np.array([]), np.array([])
X_rain_val, X_img_val, y_val = np.array([]), np.array([]), np.array([])
X_rain_test, X_img_test, y_test = np.array([]), np.array([]), np.array([])

if len(y) > 0:
    unique_labels, counts = np.unique(y, return_counts=True)
    print(f"전체 레이블 분포: {dict(zip(unique_labels, counts))}")

    if len(unique_labels) > 1: # 클래스가 2개 이상일 때 stratify 의미가 있음
        #훈련+검증 세트와 테스트 세트로 분리 (예: 80% 훈련+검증, 20% 테스트)
        X_rain_train_val, X_rain_test_temp, \
        X_img_train_val, X_img_test_temp, \
        y_train_val, y_test_temp = train_test_split(
            X_rainfall_scaled, X_image, y,
            test_size=0.2,  
            random_state=42,
            stratify=y 
        )
        X_rain_test, X_img_test, y_test = X_rain_test_temp, X_img_test_temp, y_test_temp # 최종 테스트셋 할당

        # 2. 훈련+검증 세트를 다시 훈련 세트와 검증 세트로 분리 (예: 훈련+검증의 75% 훈련, 25% 검증)
        if len(y_train_val) > 0:
            unique_labels_tv, counts_tv = np.unique(y_train_val, return_counts=True)
            if len(unique_labels_tv) > 1:
                X_rain_train_temp, X_rain_val_temp, \
                X_img_train_temp, X_img_val_temp, \
                y_train_temp, y_val_temp = train_test_split(
                    X_rain_train_val, X_img_train_val, y_train_val,
                    test_size=0.25, # 훈련+검증 세트 중 25%를 검증 세트로 (0.8 * 0.25 = 0.2)
                    random_state=42,
                    stratify=y_train_val
                )
                X_rain_train, X_img_train, y_train = X_rain_train_temp, X_img_train_temp, y_train_temp
                X_rain_val, X_img_val, y_val = X_rain_val_temp, X_img_val_temp, y_val_temp
            else: # 훈련+검증 세트에 클래스가 1개 뿐이면, 모두 훈련으로 사용하고 검증은 비움
                print("Warning: 훈련+검증 세트에 레이블이 1종류뿐입니다. 모두 훈련 데이터로 사용합니다.")
                X_rain_train, X_img_train, y_train = X_rain_train_val, X_img_train_val, y_train_val
        else: # 훈련+검증 세트가 비어있는 경우 (이론상 발생 안 함)
             print("Warning: 훈련+검증 세트가 비어있습니다.")

    else: # 전체 데이터에 클래스가 1개뿐이면
        print("Warning: 전체 데이터에 레이블이 1종류뿐입니다. stratify 없이 분할하며, 검증/테스트 의미가 적습니다.")
        # 이 경우, 대부분을 훈련으로 사용하고 일부만 테스트로 사용 (검증은 생략 가능)
        X_rain_train, X_rain_test_temp, \
        X_img_train, X_img_test_temp, \
        y_train, y_test_temp = train_test_split(
            X_rainfall_scaled, X_image, y,
            test_size=0.1, random_state=42 # 10%만 테스트로
        )
        X_rain_test, X_img_test, y_test = X_rain_test_temp, X_img_test_temp, y_test_temp
else:
    print("오류: 학습할 데이터가 없습니다.")
    exit()

print(f"훈련 데이터: 강수량 {X_rain_train.shape}, 이미지 {X_img_train.shape}, 레이블 {y_train.shape}")
if X_rain_val.shape[0] > 0:
    print(f"검증 데이터: 강수량 {X_rain_val.shape}, 이미지 {X_img_val.shape}, 레이블 {y_val.shape}")
else:
    print("검증 데이터가 생성되지 않았습니다.")
if X_rain_test.shape[0] > 0:
    print(f"테스트 데이터: 강수량 {X_rain_test.shape}, 이미지 {X_img_test.shape}, 레이블 {y_test.shape}")
else:
    print("테스트 데이터가 생성되지 않았습니다.")


# --- 4. 멀티모달 모델 구축 ---
print("모델 구축 중...")

image_input = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), name='image_input')
cnn_layer = Conv2D(32, (3, 3), activation='relu', padding='same')(image_input)
cnn_layer = MaxPooling2D((2, 2))(cnn_layer)
cnn_layer = Conv2D(64, (3, 3), activation='relu', padding='same')(cnn_layer)
cnn_layer = MaxPooling2D((2, 2))(cnn_layer)
cnn_layer = Conv2D(128, (3, 3), activation='relu', padding='same')(cnn_layer)
cnn_layer = MaxPooling2D((2, 2))(cnn_layer)
# 더 깊게 쌓거나 GlobalMaxPooling2D 고려 가능
image_flattened = Flatten()(cnn_layer)
image_features = Dense(128, activation='relu')(image_flattened)
image_features = Dropout(0.3)(image_features) # Dropout 추가

rainfall_input = Input(shape=(SEQUENCE_LENGTH,), name='rainfall_input')
# --- 강수량 처리 모델 선택 (MLP 또는 LSTM) ---
# 옵션 1: MLP
# rainfall_dense = Dense(64, activation='relu')(rainfall_input)
# rainfall_features_vec = Dense(64, activation='relu')(rainfall_dense)
# rainfall_features_vec = Dropout(0.3)(rainfall_features_vec)

# 옵션 2: LSTM (시계열 특성 더 잘 반영)
rainfall_reshaped = tf.keras.layers.Reshape((SEQUENCE_LENGTH, 1))(rainfall_input)
lstm_layer = LSTM(64, activation='relu', return_sequences=False)(rainfall_reshaped) # return_sequences=False가 기본
rainfall_features_vec = Dense(64, activation='relu')(lstm_layer)
rainfall_features_vec = Dropout(0.3)(rainfall_features_vec)
# --- 선택 끝 ---

merged_features = concatenate([image_features, rainfall_features_vec])
combined_dense = Dense(128, activation='relu')(merged_features)
combined_dropout = Dropout(0.5)(combined_dense) # 융합 후 Dropout 비율 증가
output_layer = Dense(1, activation='sigmoid', name='output_flood_prediction')(combined_dropout)

model = Model(inputs=[image_input, rainfall_input], outputs=output_layer)

optimizer = Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]) # 정밀도, 재현율 추가

model.summary()

# --- 5. 모델 학습 ---
print("모델 학습 시작...")
history = None
if X_rain_train.shape[0] > 0 and X_img_train.shape[0] > 0 and y_train.shape[0] > 0 :
    validation_data_param = None
    if X_rain_val.shape[0] > 0 and X_img_val.shape[0] > 0 and y_val.shape[0] > 0:
        validation_data_param = ([X_img_val, X_rain_val], y_val)

    # 클래스 가중치 (데이터 불균형이 심할 경우 고려)
    # from sklearn.utils import class_weight
    # class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    # class_weights_dict = dict(enumerate(class_weights))
    # print(f"클래스 가중치: {class_weights_dict}")

    history = model.fit(
        [X_img_train, X_rain_train], y_train,
        validation_data=validation_data_param,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        # class_weight=class_weights_dict # 불균형 심할 때 사용
    )
else:
    print("훈련 데이터가 없어 학습을 진행할 수 없습니다.")


# --- 6. 모델 평가 (테스트 데이터로) ---
print("모델 평가 중 (테스트 데이터)...")
if history and X_rain_test.shape[0] > 0 and X_img_test.shape[0] > 0 and y_test.shape[0] > 0:
    results = model.evaluate([X_img_test, X_rain_test], y_test, verbose=0)
    print(f"테스트 세트 결과:")
    for name, value in zip(model.metrics_names, results):
        print(f"{name}: {value:.4f}")
else:
    print("테스트 데이터가 없거나 학습이 진행되지 않아 최종 평가를 생략합니다.")
    if history and validation_data_param:
        print("검증 데이터에 대한 마지막 에포크 성능을 참고하세요.")


# --- 7. 학습 과정 시각화 (선택) ---
def plot_training_history(training_history):
    if not training_history or not training_history.history:
        print("학습 기록이 없어 시각화를 생략합니다.")
        return

    # 사용 가능한 모든 메트릭 키 가져오기
    available_metrics = [key for key in training_history.history.keys() if not key.startswith('val_')]

    num_plots = len(available_metrics)
    if num_plots == 0:
        print("시각화할 학습 메트릭이 없습니다.")
        return

    plt.figure(figsize=(6 * num_plots, 5))

    for i, metric_name in enumerate(available_metrics):
        val_metric_name = 'val_' + metric_name
        plt.subplot(1, num_plots, i + 1)

        plt.plot(training_history.history[metric_name], label=f'Training {metric_name.capitalize()}')
        if val_metric_name in training_history.history:
            plt.plot(training_history.history[val_metric_name], label=f'Validation {metric_name.capitalize()}')

        plt.title(f'{metric_name.capitalize()} Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel(metric_name.capitalize())
        plt.legend()

    plt.tight_layout()
    plt.show()

if history:
    plot_training_history(history)

# 모델 저장 (선택)
# if history:
# model.save('gangnam_flood_multimodal_predictor_5k.h5')
# print("모델이 'gangnam_flood_multimodal_predictor_5k.h5'로 저장되었습니다.")

print("스크립트 실행 완료.")