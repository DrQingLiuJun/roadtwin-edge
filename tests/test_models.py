import importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]

def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'tools'/f'{name}.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

models=load('models');export=load('export_model');changelog=load('check_changelog')

class ModelTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.root=Path(self.directory.name);self.patch=patch.object(models,'BASE',self.root);self.patch.start()
        folder=self.root/'models'/'demo';folder.mkdir(parents=True);(folder/'best.pt').write_bytes(b'weights');(folder/'model.engine').write_bytes(b'engine')
        (self.root/'runtime').mkdir();(self.root/'runtime'/'libdesktop_detector.so').touch()
        self.config={'name':'demo','pt':'models/demo/best.pt','engine':'models/demo/model.engine','library':'runtime/libdesktop_detector.so','pt_sha256':models.digest(folder/'best.pt'),'engine_sha256':models.digest(folder/'model.engine'),'input_size':[544,960],'model_classes':models.CLASSES,'wire_class_map':models.WIRE}

    def tearDown(self):self.patch.stop();self.directory.cleanup()

    def test_input_height_width_order(self):
        self.assertEqual(models.expected_shape(self.config),(1,3,544,960));self.config['input_size']=768;self.assertEqual(models.expected_shape(self.config),(1,3,768,768))

    def test_invalid_stride_rejected(self):
        self.config['input_size']=[540,960]
        with self.assertRaises(ValueError):models.expected_shape(self.config)

    def test_path_escape_rejected(self):
        path=self.root/'secret.pt';path.touch()
        with self.assertRaises(ValueError):models.local_file(str(path))

    def test_valid_dense_manifest(self):
        bindings=[{'input':True,'shape':(1,3,544,960)},{'input':False,'shape':(1,8,10710)}]
        with patch.object(models,'engine_bindings',return_value=bindings):self.assertTrue(models.verify(self.config,False)['verified'])

    def test_end_to_end_decoder_mismatch_rejected(self):
        bindings=[{'input':True,'shape':(1,3,544,960)},{'input':False,'shape':(1,300,6)}]
        with patch.object(models,'engine_bindings',return_value=bindings):
            with self.assertRaises(ValueError):models.verify(self.config,False)

    def test_engine_shape_mismatch_rejected(self):
        bindings=[{'input':True,'shape':(1,3,960,544)},{'input':False,'shape':(1,8,10710)}]
        with patch.object(models,'engine_bindings',return_value=bindings):
            with self.assertRaises(ValueError):models.verify(self.config,False)

    def test_hash_mismatch_rejected(self):
        self.config['engine_sha256']='0'*64
        with self.assertRaises(ValueError):models.verify(self.config,False)

    def test_class_reordering_rejected(self):
        self.config['model_classes']={'0':'car','1':'person','2':'cyclist','3':'truck'}
        with self.assertRaises(ValueError):models.verify(self.config,False)

    def test_atomic_config_and_changelog(self):
        readme=self.root/'README.md';readme.write_text('# Demo\n\n## 更新日志\n\n## License\n')
        models.append_log('新增测试模型。');self.assertEqual(len(changelog.entries(readme.read_text())),1)
        self.assertFalse(list(self.root.glob('.model-*')))

    def test_export_shape_validation(self):
        export.validate_shape([1,3,544,960],544,960)
        with self.assertRaises(ValueError):export.validate_shape([1,3,960,544],544,960)

if __name__=='__main__':unittest.main()
