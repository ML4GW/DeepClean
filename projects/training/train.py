from deepclean.trainer.wrapper import make_cmd_line_fn


@make_cmd_line_fn
def nds2_train(output_directory: str, verbose: bool = False, **kwargs):
    return None, None, None, None


if __name__ == "__main__":
    nds2_train()
