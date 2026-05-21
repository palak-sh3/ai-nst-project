import os
import torch
from flask import Flask, render_template, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from PIL import Image
from torchvision import transforms

from utils.utils import adaptive_instance_normalization

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

Bootstrap(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 🔥 FORCE CPU ONLY (important for Render)
device = torch.device("cpu")
torch.set_num_threads(1)

# =========================
# Forms
# =========================
class UploadForm(FlaskForm):
    content = FileField('Content Image')
    style = FileField('Style Image')
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField('Alpha', default=1.0)
    submit = SubmitField('Transfer Style')


# =========================
# Lazy Model Loading
# =========================
encoder = None
decoder = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_models():
    global encoder, decoder

    if encoder is None or decoder is None:

        # 🔥 heavy imports moved inside function
        from utils.models import VGGEncoder, Decoder

        encoder = VGGEncoder(
            os.path.join(BASE_DIR, "vgg_normalised.pth")
        )

        decoder = Decoder()

        decoder.load_state_dict(torch.load(
            os.path.join(BASE_DIR, "experiment/final_exp/decoder_final.pth"),
            map_location="cpu"
        ))

        encoder.eval()
        decoder.eval()

    return encoder, decoder


# =========================
# Utils
# =========================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def save_image(image, path):
    image = image.cpu().squeeze(0).clamp(0, 1)
    image = transforms.ToPILImage()(image)
    image.save(path)


# =========================
# Style Transfer
# =========================
def style_transfer(content_image, style_image, alpha):
    encoder, decoder = load_models()

    transform = transforms.Compose([
        transforms.Resize(256),  # 🔥 reduced for memory safety
        transforms.ToTensor()
    ])

    content = transform(content_image).unsqueeze(0)
    style = transform(style_image).unsqueeze(0)

    with torch.no_grad():
        content_feat = encoder(content, is_test=True)
        style_feat = encoder(style, is_test=True)

        t = adaptive_instance_normalization(content_feat, style_feat)
        t = alpha * t + (1 - alpha) * content_feat

        output = decoder(t)

    return output


# =========================
# Routes
# =========================
@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if form.validate_on_submit():

        # Content image
        if form.content.data:
            content_filename = secure_filename(form.content.data.filename)
            content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
            form.content.data.save(content_path)

        # Style image
        if form.style.data:
            style_filename = secure_filename(form.style.data.filename)
            style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)
            form.style.data.save(style_path)

        if content_filename and style_filename:

            try:
                content_img = Image.open(content_path).convert("RGB")
                style_img = Image.open(style_path).convert("RGB")

                alpha = float(form.alpha.data or 1.0)

                output = style_transfer(content_img, style_img, alpha)

                result_filename = "result_" + content_filename
                result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)

                save_image(output, result_path)

                result_image = result_filename

            except Exception as e:
                error = str(e)

    else:
        error = None

    return render_template(
        "index.html",
        form=form,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
        error=error
    )


# =========================
# File serving
# =========================
@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/examples/<path:filename>')
def send_example(filename):
    return send_from_directory('examples', filename)


# =========================
# MAIN (IMPORTANT: DO NOTHING HERE FOR RENDER)
# =========================
if __name__ == "__main__":
    app.run(debug=True)