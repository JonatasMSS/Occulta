import unittest

import numpy as np

from face_blur import FrameContext, Handler


class RecordingHandler(Handler):
    def __init__(self, name: str, continue_chain: bool = True) -> None:
        super().__init__()
        self.name = name
        self.continue_chain = continue_chain

    def process(self, context: FrameContext) -> bool:
        context.state.setdefault("calls", []).append((self.name, id(context)))
        return self.continue_chain


class HandlerChainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = FrameContext(
            frame_index=7,
            frame=np.zeros((2, 2, 3), dtype=np.uint8),
        )

    def test_handlers_run_in_order_with_same_context(self) -> None:
        first = RecordingHandler("first")
        second = RecordingHandler("second")
        third = RecordingHandler("third")
        first.set_next(second).set_next(third)

        result = first.handle(self.context)

        self.assertIs(result, self.context)
        self.assertEqual(
            self.context.state["calls"],
            [
                ("first", id(self.context)),
                ("second", id(self.context)),
                ("third", id(self.context)),
            ],
        )

    def test_handler_can_stop_chain(self) -> None:
        first = RecordingHandler("first", continue_chain=False)
        first.set_next(RecordingHandler("unreached"))

        first.handle(self.context)

        self.assertEqual(self.context.state["calls"], [("first", id(self.context))])


if __name__ == "__main__":
    unittest.main()
