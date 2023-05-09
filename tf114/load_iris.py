import tensorflow as tf
from sklearn.datasets import load_iris

# Iris 데이터셋 로드
iris = load_iris()

# 특징 데이터
X = iris.data

# 타깃 데이터
y = iris.target

# TensorFlow 그래프 생성
graph = tf.Graph()

with graph.as_default():
    # 입력 플레이스홀더
    X_placeholder = tf.placeholder(tf.float32, shape=(None, 4))

    # 타깃 플레이스홀더
    y_placeholder = tf.placeholder(tf.int32, shape=(None))

    # 데이터셋 생성
    dataset = tf.data.Dataset.from_tensor_slices((X_placeholder, y_placeholder))
    dataset = dataset.shuffle(buffer_size=100).batch(32).repeat()

    # Iterator 생성
    iterator = dataset.make_initializable_iterator()

    # 다음 배치를 가져올 연산
    next_batch = iterator.get_next()

# TensorFlow 세션 실행
with tf.Session(graph=graph) as sess:
    # Iterator 초기화
    sess.run(iterator.initializer, feed_dict={X_placeholder: X, y_placeholder: y})

    # 데이터셋에서 배치 가져오기
    for _ in range(10):
        batch_X, batch_y = sess.run(next_batch)
        print("배치 X:", batch_X)
        print("배치 y:", batch_y)
        print("---------")