from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from PIL import Image
import torch
from torch import nn
from torchvision import transforms, models
import os
from storage import attribute_names

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Константы
MODEL_PATH = "personality_net.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224  # Размер изображения для модели

# Подготовка трансформации
test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Модель
class PersonalityNet(nn.Module):
    def __init__(self, num_classes):
        super(PersonalityNet, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return torch.sigmoid(self.model(x))

# Загрузка модели
model = PersonalityNet(40).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Функция для анализа изображения
def analyze_photo_with_model(photo_path):
    try:
        # Загрузка изображения
        image = Image.open(photo_path).convert("RGB")
        input_tensor = test_transform(image).unsqueeze(0).to(DEVICE)

        # Прогнозирование
        with torch.no_grad():
            prediction = model(input_tensor).cpu().numpy().flatten()

        # Преобразование в словарь
        decoded_predictions = {attr: float(prob) for attr, prob in zip(attribute_names, prediction)}

        return decoded_predictions

    except Exception as e:
        print(f"Ошибка в анализе изображения: {e}")
        return None

# Проверка расширений файлов
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Главная страница
@app.route('/')
def index():
    if 'username' in session:
        return render_template('index.html')
    return redirect(url_for('login'))

# Страница регистрации
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Тут нужно добавить сохранение пользователя в базу данных
        session['username'] = username
        return redirect(url_for('index'))
    return render_template('register.html')

# Страница авторизации
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Тут нужно добавить проверку пользователя из базы данных
        session['username'] = username
        return redirect(url_for('index'))
    return render_template('login.html')

# Страница загрузки фотографий
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        file1 = request.files['file1']
        file2 = request.files['file2']
        if file1 and allowed_file(file1.filename) and file2 and allowed_file(file2.filename):
            filename1 = secure_filename(file1.filename)
            filename2 = secure_filename(file2.filename)
            file1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename1))
            file2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename2))

            # Анализируем изображения
            result1 = analyze_photo_with_model(os.path.join(app.config['UPLOAD_FOLDER'], filename1))
            result2 = analyze_photo_with_model(os.path.join(app.config['UPLOAD_FOLDER'], filename2))

            # Передаем изображения и результаты в шаблон
            return render_template('result.html',
                                   result1=result1,
                                   result2=result2,
                                   img1_filename=filename1,
                                   img2_filename=filename2)

    return render_template('upload.html')


# Выход из аккаунта
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
