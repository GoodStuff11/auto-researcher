import textwrap
import pytest


@pytest.fixture
def hub_root(tmp_path):
    (tmp_path / "project.yaml").write_text(textwrap.dedent("""\
        name: demo
        people:
          alice: {name: Alice}
          bob: {name: Bob}
        series:
          group: {attendees: [alice, bob]}
        roots:
          CODE: /nowhere/code
        """))
    (tmp_path / "hub.local.yaml").write_text("me: alice\nroots:\n  CODE: /here/code\n")
    return tmp_path
