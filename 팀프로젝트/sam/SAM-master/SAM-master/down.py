import urllib.request

url = 'https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat'
urllib.request.urlretrieve(url, 'shape_predictor_68_face_landmarks.dat')