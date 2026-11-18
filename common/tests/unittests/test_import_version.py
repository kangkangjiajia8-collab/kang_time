import autoapex.common


def test_import_version():
    assert isinstance(autoapex.common.__version__, str)
    assert len(autoapex.common.__version__) != 0
