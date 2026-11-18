import autoapex.core


def test_import_version():
    assert isinstance(autoapex.core.__version__, str)
    assert len(autoapex.core.__version__) != 0
