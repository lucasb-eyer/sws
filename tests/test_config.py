import pytest
from sws import Config

def test_construct_kwargs():
    c = Config(
        an_int=1,
        a_string="hi",
        a_float=0.3,
    ).finalize()
    assert c.an_int == 1 and c.a_string == "hi" and c.a_float == 0.3


def test_construct_attrs():
    c = Config()
    c.an_int = 1
    c.a_string = "hi"
    c.a_float = 0.3
    c = c.finalize()
    assert c.an_int == 1 and c.a_string == "hi" and c.a_float == 0.3


def test_construct_items():
    c = Config()
    c["an_int"] = 1
    c["a_string"] = "hi"
    c["a_float"] = 0.3
    c = c.finalize()
    assert c.an_int == 1 and c.a_string == "hi" and c.a_float == 0.3


def test_to_dict():
    d = dict(
        an_int=1,
        a_string="hi",
        a_float=0.3,
    )
    assert Config(**d).to_dict() == d
    assert Config(**d).finalize().to_dict() == d


def test_update_fields():
    c = Config(an_int=1)
    c.an_int = 3
    c["an_int"] = 5
    f = c.finalize()
    assert f.an_int == 5


def test_odd_names():
    c = Config()
    c["1odd-name yay!"] = "why not?"
    f = c.finalize()
    assert f["1odd-name yay!"] == "why not?"


def test_nested_kwargs():
    c = Config(lr=0.1, model=dict(width=128, depth=4))

    # Check that we can add onto existing group:
    c.model.expand = 2
    c["model"]["head_dim"] = 32

    # Check that we can overwrite/modify existing nests,
    # including with mixed [] and . access:
    c["model"].depth = 8
    c.model["width"] = 64

    c = c.finalize()

    assert c.lr == 0.1
    assert c.model.width == 64
    assert c.model.depth == 8
    assert c.model.expand == 2
    assert c.model.head_dim == 32


def test_flat_set_and_get():
    c = Config(lr=0.01)
    c['model.width'] = 128
    c['model.depth'] = 4
    c = c.finalize()
    assert c.get("model.width") == c.model.width == 128
    assert c.get("model.depth") == c.model.depth == 4

    c = Config(lr=0.01)
    c.model.width = 128
    c.model.depth = 4
    c = c.finalize()
    assert c.get("model.width") == c.model.width == 128
    assert c.get("model.depth") == c.model.depth == 4

    # Also check that get with default works:
    assert c.get("model.width", 3) == 128
    assert c.get("model.bar.baz", 3) == 3
    assert c.get("lol") == None


def test_reading_leaves_is_disallowed():
    c = Config(lr=0.1, model=dict(width=128))
    # Direct attribute and index reads of leaves are disallowed pre-finalize
    with pytest.raises(TypeError):
        _ = c.lr
    with pytest.raises(TypeError):
        _ = c["lr"]
    # Nested leaf reads (via view or flat key) are also disallowed
    with pytest.raises(TypeError):
        _ = c.model.width
    with pytest.raises(TypeError):
        _ = c["model.width"]
    # Creating a new leaf and then reading it is also disallowed
    c.model.depth = 4
    with pytest.raises(TypeError):
        _ = c.model.depth


def test_shadowing_rules():
    # Cannot set leaf at group root when group exists
    c = Config(model=dict(width=128))
    with pytest.raises(ValueError):
        c.model = 5

    # Cannot set group when a leaf exists at exact root
    c2 = Config(lr=3)
    with pytest.raises(TypeError):
        c2.lr.schedule = 128


def test_delete_group_and_leaf():
    c = Config(model={'width': 128, 'depth': 4}, lr=0.1)

    # Deleting a leaf works
    del c['model.width']
    assert 'model.width' not in c
    assert 'model.depth' in c
    assert 'lr' in c

    # Deleting a whole group works too
    del c['model']
    assert 'model' not in c
    assert 'model.depth' not in c
    with pytest.raises(KeyError):
        del c['model.depth']
    with pytest.raises(KeyError):
        del c['model']


def test_dict_conversion():
    d = {'lr': 0.1, 'model': {'width': 128, 'depth': 4}}
    c = Config(**d).finalize()
    assert c.to_dict() == d
    assert c.model.to_dict() == d['model']

    d = {'lr': 0.1, 'model.width': 128, 'model.depth': 4}
    c = Config(**d).finalize()
    assert c.to_flat_dict() == d
    assert c.model.to_flat_dict() == {'width': 128, 'depth': 4}
