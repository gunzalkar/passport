import os
import traceback
import cv2
from passporteye import read_mrz
from mrz.checker.td3 import TD3CodeChecker


def calculate_check_digit(data):
    weight = [7, 3, 1]
    total = 0
    for i, char in enumerate(data):
        if char.isdigit():
            value = int(char)
        elif char.isalpha():
            value = ord(char) - ord('A') + 10
        elif char == '<':
            value = 0
        else:
            return False
        total += value * weight[i % 3]
    return str(total % 10)


def validate_mrz_field(mrz_field):
    data = mrz_field[:-1]
    expected_check_digit = mrz_field[-1]
    calculated_check_digit = calculate_check_digit(data)
    if calculated_check_digit:
        return calculated_check_digit == expected_check_digit
    return False


def reformat_date(date_str):
    if len(date_str) != 6:
        raise ValueError("Date string must be in YYMMDD format")
    day = date_str[4:6]
    month = date_str[2:4]
    year = date_str[0:2]
    year = '19' + year if int(year) > 50 else '20' + year
    return f"{day}.{month}.{year}"


def crop_mrz(image):
    height, width = image.shape[:2]
    mrz_height = int(height * 0.2)
    cropped_image = image[height - mrz_height:height, 0:width]
    return cropped_image


def resize_image(image, scale_percent=420):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized_image


def detect_and_crop_face(image, padding_percent=0.2):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        x_padding = int(w * padding_percent)
        y_padding = int(h * padding_percent)
        x_start = max(x - x_padding, 0)
        y_start = max(y - y_padding, 0)
        x_end = min(x + w + x_padding, image.shape[1])
        y_end = min(y + h + y_padding, image.shape[0])
        face_image = image[y_start:y_end, x_start:x_end]
        return face_image
    return None


def crop_to_content(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cropped_image = image[y:y + h, x:x + w]
        return cropped_image
    return image


def improve_mrz_accuracy_with_passporteye(image, resized_image_path):
    image = crop_to_content(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = crop_mrz(gray_image)
    resized_image = resize_image(gray_image)

    # Save the resized image
    cv2.imwrite(resized_image_path, resized_image)

    # Read MRZ from the saved resized image
    # print(resized_image_path)
    mrz = read_mrz(resized_image_path)
    # print(mrz)
    mrz_data = mrz.to_dict()
    # for key, value in mrz_data.items():
    #     print(f"{key}: {value}")
    # if not mrz:
    #     return "", None, gray_image
    os.remove(resized_image_path)
    # mrz_data = mrz.to_dict()
    # text = f"{mrz_data['mrz1']}\n{mrz_data['mrz2']}".replace(' ', '').upper()
    face_img = detect_and_crop_face(image)
    return mrz_data['raw_text'], face_img, gray_image


def get_info(img_path):
    try:
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not read image, please check the path.")

        resized_image_path = 'resized_image.jpg'  # Path to save the resized image
        text_data, face_img, mrz_code_img = improve_mrz_accuracy_with_passporteye(image, resized_image_path)
        # print(text_data.split("\n"))
        if text_data:
            final_mrz_lines = text_data
            td3_check = TD3CodeChecker(final_mrz_lines, check_expiry=True)
            fields = td3_check.fields()
            invalid_fields = [field[0] for field in td3_check.report.falses] if td3_check.report.falses else []
            birth_date = reformat_date(fields.birth_date)
            expiry_date = reformat_date(fields.expiry_date)

            def check_falses(key_words):
                for invalid_field in invalid_fields:
                    if key_words in invalid_field:
                        return False
                return True

            fields_dict = {
                'full_mrz': {'value': final_mrz_lines, 'status': check_falses('final hash')},
                'mrz_birth_date': {'value': birth_date, 'status': check_falses('birth')},
                'mrz_cd_birth_date': {'value': fields.birth_date_hash, 'status': check_falses('birth date hash')},
                'mrz_cd_composite': {'value': fields.final_hash, 'status': True},
                'mrz_cd_expiry_date': {'value': fields.expiry_date_hash, 'status': check_falses('expiry date hash')},
                'mrz_cd_number': {'value': fields.document_number_hash, 'status': check_falses('document number hash')},
                'mrz_cd_opt_data_2': {'value': fields.optional_data_hash, 'status': check_falses('optional data hash')},
                'mrz_doc_type_code': {'value': fields.document_type, 'status': check_falses('document type')},
                'mrz_expiry_date': {'value': expiry_date, 'status': check_falses('expiry date')},
                'mrz_gender': {'value': fields.sex, 'status': check_falses('sex')},
                'mrz_issuer': {'value': fields.country, 'status': check_falses('nationality')},
                'mrz_last_name': {'value': fields.surname, 'status': True},
                'mrz_line1': {'value': final_mrz_lines.split('\n')[0], 'status': check_falses('final hash')},
                'mrz_line2': {'value': final_mrz_lines.split('\n')[1], 'status': check_falses('final hash')},
                'mrz_name': {'value': fields.name, 'status': True},
                'mrz_nationality': {'value': fields.nationality, 'status': check_falses('nationality')},
                'mrz_number': {'value': fields.document_number, 'status': check_falses('document number')},
                'mrz_opt_data_2': {'value': fields.optional_data, 'status': check_falses('optional data')},
            }
            return fields_dict, face_img, mrz_code_img
        else:
            return "Couldn't extract data from image. Try more clear image."
    except Exception as e:
        print(traceback.print_exc())
        print(e)
        return "Failed to process image. Please try again"

# Example usage
# if __name__ == "__main__":
#     img_path = 'actual.jpg'
#     result = get_info(img_path)
#     print(result)
