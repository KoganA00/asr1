# Don't forget to support cases when target_text == ''
import editdistance

def calc_cer(target_text, predicted_text) -> float:
    # TODO: your code here
    if len(target_text) == 0:
        return float(len(predicted_text) != 0)
    return editdistance.eval(target_text, predicted_text) / len(target_text)
    raise NotImplementedError()


def calc_wer(target_text, predicted_text) -> float:
    # TODO: your code here
    target_words = target_text.split()
    predicted_words = predicted_text.split()

    if len(target_words) == 0:
        return float(len(predicted_words) != 0)
    return editdistance.eval(target_words, predicted_words) / len(target_words)

    raise NotImplementedError()
