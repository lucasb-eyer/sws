import pytest
import sws


def test_access_simple():
    c = sws.Config(
        an_int=1,
        a_string="hi",
        a_float=0.3,
    )
    # Reads from Config leaves are disallowed; use finalize() to read
    with pytest.raises(TypeError):
        _ = c.an_int
    with pytest.raises(TypeError):
        _ = c["an_int"]
    f = c.finalize()
    assert f.an_int == 1 and f.a_string == "hi" and f.a_float == 0.3


def test_to_dict():
    d = dict(
        an_int=1,
        a_string="hi",
        a_float=0.3,
    )
    c = sws.Config(**d)
    assert c.to_dict() == d


def test_add_update_fields():
    c = sws.Config(an_int=1)
    c.an_int = 3
    c.new_field = "Hello!"
    c["1odd-name"] = "why not?"
    with pytest.raises(TypeError):
        _ = c.an_int
    with pytest.raises(TypeError):
        _ = c["new_field"]
    f = c.finalize()
    assert f.an_int == 3 and f.new_field == "Hello!" and f.to_flat_dict()["1odd-name"] == "why not?"


def test_nesting():
    c = sws.Config(lr=0.1, model=dict(width=128, depth=4))

    # Reading leaves from Config is disallowed
    with pytest.raises(TypeError):
        _ = c.model.width
    with pytest.raises(TypeError):
        _ = c["model.width"]
    # Group views work for nested writes
    c.model.expand = 2
    with pytest.raises(TypeError):
        _ = c.model.expand

    # Accessing attribute of a leaf raises TypeError; missing group raises AttributeError
    with pytest.raises(TypeError):
        c.lr.something = 3
    with pytest.raises(AttributeError):
        c.inexistant.something = 3

    assert "model" in c
    assert "model.width" in c

    # get still works as a convenience on Config
    assert c.get("model.width", 3) == 128
    assert c.get("model.bar.baz", 3) == 3
    assert c.get("lol") == None


def test_flat_set_and_get():
    c = sws.Config()
    c['lr'] = 0.01
    c['model.width'] = 128
    c['model.depth'] = 4

    with pytest.raises(TypeError):
        _ = c['lr']
    with pytest.raises(TypeError):
        _ = c['model.width']
    with pytest.raises(TypeError):
        _ = c.model.width
    assert 'model.width' in c
    assert 'model' in c
    assert len(c) == 2  # lr, model


def test_group_view_shared_store():
    c = sws.Config(model={'width': 128})
    v = c['model']
    assert isinstance(v, sws.Config)
    v['depth'] = 6
    assert 'model.depth' in c
    v.width = 64
    assert 'model.width' in c


def test_shadowing_rules():
    c = sws.Config()
    c['model.width'] = 128
    # Cannot set leaf at group root when group exists
    with pytest.raises(ValueError):
        c['model'] = 5

    c2 = sws.Config()
    c2['model'] = 3
    # Cannot set group when a leaf exists at exact root
    with pytest.raises(ValueError):
        c2['model'] = {'depth': 4}


def test_delete_group_and_leaf():
    c = sws.Config(model={'width': 128, 'depth': 4}, lr=0.1)
    del c['model.width']
    assert 'model.width' not in c
    assert 'model.depth' in c
    del c['model']  # delete whole group
    assert 'model' not in c
    assert 'model.depth' not in c
    with pytest.raises(KeyError):
        del c['model']


def test_iteration_and_keys():
    c = sws.Config(lr=0.1, model={'width': 128, 'depth': 4})
    ks = set(c)
    assert ks == {'lr', 'model'}
    # iterating yields keys; indexing a group yields a view; leaf reads are disallowed
    for k in c:
        if k == 'model':
            assert isinstance(c[k], sws.Config)
        if k == 'lr':
            with pytest.raises(TypeError):
                _ = c[k]


def test_to_flat_and_to_dict():
    c = sws.Config(model={'width': 128, 'depth': 4}, lr=0.1)
    assert c.to_dict() == {'model': {'width': 128, 'depth': 4}, 'lr': 0.1}
    flat = c.to_flat_dict()
    assert flat['model.width'] == 128 and flat['model.depth'] == 4 and flat['lr'] == 0.1
    mv = c['model']
    mflat = mv.to_flat_dict()
    assert mflat == {'width': 128, 'depth': 4}
