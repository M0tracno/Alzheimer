"""Run with: python -m unittest discover tests"""
import io
import os
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import Config  # noqa: E402

TMP = tempfile.mkdtemp()
Config.DATABASE = os.path.join(TMP, 'test.db')
Config.UPLOAD_FOLDER = TMP

import app as webapp  # noqa: E402


class FakeModel:
    """Always answers class index 2, which must be shown as 'No Impairment'."""
    def predict(self, batch, verbose=0):
        assert batch.shape == (1, *Config.IMAGE_SIZE, 3)
        return np.array([[0.05, 0.05, 0.85, 0.05]])


def png_bytes():
    buf = io.BytesIO()
    Image.new('L', (200, 180), 90).save(buf, 'PNG')
    buf.seek(0)
    return buf


class AppTest(unittest.TestCase):
    def setUp(self):
        webapp._model = FakeModel()
        webapp.app.config['TESTING'] = True
        self.client = webapp.app.test_client()

    def register_and_login(self, name):
        self.client.post('/register', data={'username': name, 'email': f'{name}@x.org',
                                            'password': 'secret1', 'confirm_password': 'secret1'})
        return self.client.post('/login', data={'username': name, 'password': 'secret1'})

    def admin_login(self):
        return self.client.post('/admin/login', data={'username': 'admin', 'password': Config.ADMIN_PASSWORD})

    def test_public_pages_render(self):
        for url in ['/', '/login', '/register', '/admin/login']:
            self.assertEqual(self.client.get(url).status_code, 200, url)

    def test_upload_predicts_with_correct_label_and_dashboard_renders(self):
        self.assertEqual(self.register_and_login('alice').status_code, 302)
        r = self.client.post('/predict', data={'file': (png_bytes(), '../../evil scan.png')},
                             content_type='multipart/form-data', follow_redirects=True)
        self.assertEqual(r.status_code, 200)
        self.assertIn(b'No Impairment', r.data)
        saved = [f for f in os.listdir(TMP) if f.endswith('.png')]
        self.assertTrue(saved and all('..' not in f and '/' not in f for f in saved))

    def test_rejects_non_image_extension(self):
        self.register_and_login('bob')
        r = self.client.post('/predict', data={'file': (io.BytesIO(b'x'), 'notes.txt')},
                             content_type='multipart/form-data', follow_redirects=True)
        self.assertIn(b'JPG or PNG', r.data)

    def test_admin_pages_and_post_only_delete(self):
        self.register_and_login('carol')
        self.client.get('/logout')
        self.assertEqual(self.admin_login().status_code, 302)
        for url in ['/admin/dashboard', '/admin/manage_users', '/admin/add_patient', '/admin/generate_report']:
            self.assertEqual(self.client.get(url).status_code, 200, url)
        db = webapp.sqlite3.connect(Config.DATABASE)
        uid = db.execute("SELECT id FROM users WHERE username = 'carol'").fetchone()[0]
        self.assertEqual(self.client.get(f'/admin/delete_user/{uid}').status_code, 405)
        self.client.post(f'/admin/delete_user/{uid}')
        self.assertIsNone(db.execute("SELECT id FROM users WHERE username = 'carol'").fetchone())
        db.close()

    def test_admin_pages_need_admin(self):
        self.register_and_login('dave')
        self.assertEqual(self.client.get('/admin/dashboard').status_code, 302)


@unittest.skipUnless(os.path.exists(Config.MODEL_PATH) and os.path.isdir(os.path.join(ROOT, 'dataset')),
                     'needs models/resnet50_alzheimer.keras and the dataset')
class RealModelLabelOrderTest(unittest.TestCase):
    def test_each_class_is_mostly_predicted_as_itself(self):
        webapp._model = None
        model = webapp.get_model()
        test_dir = os.path.join(ROOT, 'dataset', 'Combined Dataset', 'test')
        for folder in sorted(os.listdir(test_dir)):
            # With a wrong label order every class scores near 0; the real model scores ~80-93% per class.
            files = sorted(os.listdir(os.path.join(test_dir, folder)))[:20]
            batch = np.concatenate([webapp.preprocess(os.path.join(test_dir, folder, f)) for f in files])
            labels = [webapp.CLASS_NAMES[i] for i in model.predict(batch, verbose=0).argmax(1)]
            self.assertGreater(labels.count(folder), len(files) // 2, f'{folder}: {labels}')


if __name__ == '__main__':
    unittest.main()
