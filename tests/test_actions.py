import pytest

from labquake_explorer.ui import actions as A
from labquake_explorer.ui.context import EVENT, RUN, RUN_ARRAY


@pytest.fixture
def isolated_registry():
    saved = A.all_actions()
    A.clear_registry()
    yield
    A.clear_registry()
    for action in saved:
        A.add_action(action)


def test_register_action_orders_by_order_then_registration(isolated_registry):
    @A.register_action("B", kinds=[EVENT], order=20)
    def b(app, ctx):
        return "b"

    @A.register_action("A", kinds=[EVENT], order=10)
    def a(app, ctx):
        return "a"

    @A.register_action("C", kinds=[EVENT], order=20)
    def c(app, ctx):
        return "c"

    assert [x.label for x in A.actions_for(EVENT)] == ["A", "B", "C"]
    assert A.actions_for(RUN) == []
    assert A.actions_for(EVENT)[0].run(None, None) == "a"


def test_register_view_builds_from_context(isolated_registry):
    with pytest.raises(TypeError):
        @A.register_view("X", kinds=[EVENT])
        class NoFactory:
            pass

    made = []

    @A.register_view("Open", kinds=[EVENT, RUN])
    class V:
        @classmethod
        def from_context(cls, app, ctx):
            made.append((app, ctx))
            return "view"

    assert V.menu_label == "Open"
    assert V.menu_kinds == frozenset({EVENT, RUN})
    assert A.actions_for(RUN)[0].run("app", "ctx") == "view"
    assert made == [("app", "ctx")]


def test_unknown_kind_rejected_and_add_action_replaces_by_name(isolated_registry):
    with pytest.raises(ValueError):
        A.register_action("bad", kinds=["nope"])
    with pytest.raises(ValueError):
        A.register_action("bad", kinds=[])

    def f(app, ctx):
        return 1

    A.add_action(A.Action("one", frozenset([RUN_ARRAY]), f, 5, "same"))
    A.add_action(A.Action("two", frozenset([RUN_ARRAY]), f, 5, "same"))
    assert [a.label for a in A.actions_for(RUN_ARRAY)] == ["two"]
