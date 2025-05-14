from llmtuner.tuner.core.arc_tokenizer import ARCTokenizer


def test_arc_tokenizer_vocab_size(model_to_load: str = "../configs/v2"):
    """Tests the vocab size of the ARCTokenizer."""
    tokenizer = ARCTokenizer.from_pretrained(model_to_load)
    vocab_size = len(tokenizer.get_vocab())
    assert vocab_size == 22, f"Expected vocab size of 22, but got {vocab_size}."


def test_arc_tokenizer_encode_decode(model_to_load: str = "../configs/v2"):
    """Tests the encode and decode methods of the ARCTokenizer."""
    tokenizer = ARCTokenizer.from_pretrained(model_to_load)

    sample_task_str = "212<arc_grid_endx><arc_pad><arc_sep_row><arc_sep_grid>1212<arc_grid_endx><arc_sep_row>" + \
                      "020<arc_grid_endx><arc_pad><arc_sep_row><arc_sep_grid>2020<arc_grid_endx><arc_sep_row>" + \
                      "212<arc_grid_endx><arc_pad><arc_sep_row><arc_sep_grid>1212<arc_grid_endx><arc_sep_row>" + \
                      "<arc_grid_endy><arc_grid_endy><arc_grid_endy><arc_grid_endxy><arc_grid_endy><arc_sep_row><arc_sep_grid>" + \
                      "2020<arc_grid_endx><arc_sep_row>" + \
                      "<arc_pad><arc_pad><arc_pad><arc_grid_endx><arc_pad><arc_sep_row><arc_sep_grid>" + \
                      "<arc_grid_endy><arc_grid_endy><arc_grid_endy><arc_grid_endy><arc_grid_endxy><arc_sep_row>" + \
                      "<arc_sep_example>" * 13
    encoded_task = tokenizer.encode(sample_task_str)
    decoded_task = tokenizer.decode(encoded_task)
    assert decoded_task == sample_task_str, \
        f"Decoded text '{decoded_task}' does not match original text '{sample_task_str}'."
