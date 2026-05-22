import os
from pathlib import Path

import torch
from flask import Flask, render_template, request, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from PIL import Image
from torchvision import transforms

# Import AdaIN files
from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization


# -------------------- Flask App --------------------

app = Flask(__name__)

app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

Bootstrap(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# -------------------- Form --------------------

class UploadForm(FlaskForm):
    content = FileField('Content Image')
    style = FileField('Style Image')

    content_path = HiddenField()
    style_path = HiddenField()

    alpha = FloatField('Alpha', default=1.0)

    submit = SubmitField('Transfer Style')


# -------------------- Device --------------------

device = torch.device("cpu")


# -------------------- Paths --------------------

BASE_DIR = Path(__file__).resolve().parent

vgg_path = BASE_DIR / "vgg_normalised.pth"
decoder_path = BASE_DIR / "decoder_final.pth"


# -------------------- Load Models --------------------

encoder = VGGEncoder(vgg_path).to(device)

decoder = Decoder().to(device)
decoder.load_state_dict(
    torch.load(decoder_path, map_location=device)
)

encoder.eval()
decoder.eval()

# Disable gradients completely
for param in encoder.parameters():
    param.requires_grad = False

for param in decoder.parameters():
    param.requires_grad = False


# -------------------- Helpers --------------------

def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
    )


def style_transfer(content_image, style_image, encoder, decoder, alpha, device):

    transform = transforms.Compose([
        transforms.Resize((128, 128)),   # memory optimization
        transforms.ToTensor()
    ])

    content = transform(content_image).unsqueeze(0).to(device)
    style = transform(style_image).unsqueeze(0).to(device)

    with torch.no_grad():

        content_features = encoder(content, is_test=True)
        style_features = encoder(style, is_test=True)

        stylized_features = adaptive_instance_normalization(
            content_features,
            style_features
        )

        stylized_features = (
            alpha * stylized_features +
            (1 - alpha) * content_features
        )

        output = decoder(stylized_features)

    return output


def save_image(image_tensor, path):

    image_tensor = image_tensor.cpu().clone()
    image_tensor = image_tensor.squeeze(0)
    image_tensor = image_tensor.clamp(0, 1)

    image = transforms.ToPILImage()(image_tensor)

    image.save(path)


# -------------------- Routes --------------------

@app.route('/', methods=['GET', 'POST'])
def index():

    form = UploadForm()

    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if form.validate_on_submit():

        # ---------- Content Image ----------

        if form.content.data and form.content.data.filename:

            if allowed_file(form.content.data.filename):

                content_filename = secure_filename(
                    form.content.data.filename
                )

                content_path = os.path.join(
                    app.config['UPLOAD_FOLDER'],
                    content_filename
                )

                form.content.data.save(content_path)

                form.content_path.data = content_filename

        else:
            content_filename = form.content_path.data

        # ---------- Style Image ----------

        if form.style.data and form.style.data.filename:

            if allowed_file(form.style.data.filename):

                style_filename = secure_filename(
                    form.style.data.filename
                )

                style_path = os.path.join(
                    app.config['UPLOAD_FOLDER'],
                    style_filename
                )

                form.style.data.save(style_path)

                form.style_path.data = style_filename

        else:
            style_filename = form.style_path.data

        # ---------- Style Transfer ----------

        if content_filename and style_filename:

            try:

                content_path = os.path.join(
                    app.config['UPLOAD_FOLDER'],
                    content_filename
                )

                style_path = os.path.join(
                    app.config['UPLOAD_FOLDER'],
                    style_filename
                )

                content_image = Image.open(
                    content_path
                ).convert('RGB')

                style_image = Image.open(
                    style_path
                ).convert('RGB')

                # Prevent huge uploads
                content_image.thumbnail((256, 256))
                style_image.thumbnail((256, 256))

                alpha = float(form.alpha.data)

                stylized_image = style_transfer(
                    content_image,
                    style_image,
                    encoder,
                    decoder,
                    alpha,
                    device
                )

                result_filename = (
                    "stylized_" + content_filename
                )

                result_path = os.path.join(
                    app.config['UPLOAD_FOLDER'],
                    result_filename
                )

                save_image(stylized_image, result_path)

                result_image = result_filename

                # Cleanup memory
                del stylized_image

            except Exception as e:
                error = str(e)

    elif request.method == 'POST':

        if not content_filename:
            error = "Please upload content image"

        elif not style_filename:
            error = "Please upload style image"

    return render_template(
        'index.html',
        form=form,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
        error=error
    )


@app.route('/uploads/<filename>')
def send_image(filename):

    return send_from_directory(
        app.config['UPLOAD_FOLDER'],
        filename
    )


@app.route('/examples/<path:filename>')
def send_example(filename):

    return send_from_directory(
        'examples',
        filename
    )


# -------------------- Main --------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port
    )