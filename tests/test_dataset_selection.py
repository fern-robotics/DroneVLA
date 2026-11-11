import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.dataset.select import EpisodeMeta, assign_splits, intent_tag, select_stratified, template_of


def meta(i, tag, frames, template):
    return EpisodeMeta(str(i), f"Move {i}", tag, template, frames, "short" if frames < 20 else ("medium" if frames <= 40 else "long"), "s0")


class SelectionTest(unittest.TestCase):
    def test_tags_and_templates(self):
        self.assertEqual(intent_tag("Move to the tree on the front side"), "approach")
        self.assertEqual(intent_tag("Orbit the dog to the left"), "orbit")
        self.assertEqual(template_of("Move 7.0 metres to the left"), "move <n> metres to the left")

    def test_stratified_and_disjoint(self):
        metas = [meta(i, ["move", "orbit", "pass"][i % 3], [10, 30, 50][i % 3], f"t{i % 5}") for i in range(90)]
        chosen = select_stratified(metas, 0.1, seed=0)
        self.assertEqual(len(chosen), 9)
        splits, flags, ood = assign_splits(chosen, seed=0)
        self.assertEqual(set(splits), {m.id for m in chosen})
        self.assertTrue(set(ood))
        self.assertTrue(any(flags.values()))
        train_templates = {m.template for m in chosen if splits[m.id] == "train"}
        ood_templates = {m.template for m in chosen if flags[m.id]}
        self.assertFalse(train_templates & ood_templates)


if __name__ == "__main__":
    unittest.main()
