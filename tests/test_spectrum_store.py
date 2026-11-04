"""Named result sets of a record's spectrum store."""
import pytest

from labquake_explorer.data import spectrum_store as st


def test_set_name_validation():
    assert st.set_name("  EGF ") == "EGF"
    for bad in ("", "   ", None, "a/b"):
        with pytest.raises(ValueError):
            st.set_name(bad)


def test_version_1_store_reads_as_the_default_set():
    v1 = {"version": 1, "channels": {"ch0": {"channel": 0}}}
    assert st.store_sets(v1) == {"default": v1}
    assert st.set_names(v1) == ["default"]
    assert st.set_channels(v1) is v1["channels"] and st.set_channels(v1, "EGF") is None
    assert st.store_sets("junk") == {} and st.set_names(None) == [] and st.set_channels({}, "default") is None
    v2 = {"version": 2, "sets": {"EGF": {"channels": {}}, "BAC": {"channels": {}}, "junk": 3}}
    assert st.set_names(v2) == ["BAC", "EGF"]


def test_upgrade_and_ensure_set():
    v1 = {"version": 1, "channels": {"ch0": {"channel": 0}}}
    assert st.upgrade_store(v1)
    assert v1 == {"version": 2, "sets": {"default": {"channels": {"ch0": {"channel": 0}}}}}
    assert not st.upgrade_store(v1)
    egf = st.ensure_set(v1, "EGF")
    assert egf == {"channels": {}} and v1["sets"]["EGF"] is egf
    assert st.set_names(v1) == ["default", "EGF"]
    assert st.set_channels(v1, "default") == {"ch0": {"channel": 0}}
    # a list of channels (older HDF5 files) is kept as it is in the default set
    v1_list = {"version": 1, "channels": [{"channel": 0}]}
    st.upgrade_store(v1_list)
    assert st.set_channels(v1_list) == [{"channel": 0}]
    assert st.ensure_set({}, "x") == {"channels": {}}
    assert not st.upgrade_store("junk")
