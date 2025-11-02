from flask import Blueprint, request, jsonify
from flask_mail import Message
from extensions import mail

others_bp = Blueprint('others', __name__)

@others_bp.route('/collaborate', methods=['POST'])
def collaborate():
    """
    Endpoint for the 'Collaborate with Us' form.
    """
    try:
        data = request.get_json()
        name = data.get('name')
        device_model = data.get('deviceModel')
        image_count = data.get('imageCount')
        email = data.get('email')
        img_format = data.get('format')

        if not all([name, device_model, image_count, email, img_format]):
            return jsonify({'error': 'Missing required form fields.'}), 400

        return jsonify({
            'message': 'Form submitted successfully!',
            'received': {
                'name': name,
                'deviceModel': device_model,
                'imageCount': image_count,
                'email': email,
                'format': img_format
            }
        }), 200

    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


@others_bp.route('/contact', methods=['POST'])
def contact():
    """
    Endpoint for the 'Contact Us' form.
    Sends email with provided data to fotoforensics3@gmail.com.
    """
    try:
        data = request.get_json()
        name = data.get('name')
        email = data.get('email')
        subject = data.get('subject')
        message = data.get('message')

        if not all([name, email, subject, message]):
            return jsonify({'error': 'Missing required fields in the contact form.'}), 400

        # ✉️ Utworzenie wiadomości e-mail
        msg = Message(
            subject=f"[Foto Forensics Contact] {subject}",
            recipients=['fotoforensics3@gmail.com'],
            body=(
                f"📩 New message from the contact form:\n\n"
                f"👤 Name: {name}\n"
                f"📧 Email: {email}\n"
                f"📝 Subject: {subject}\n\n"
                f"💬 Message:\n{message}\n"
            )
        )

        # Wysłanie wiadomości
        mail.send(msg)

        return jsonify({'message': 'Your message has been sent successfully!'}), 200

    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500
