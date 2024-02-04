import sacrebleu

ref="You can come back any time as our chat service window is open 24/7"
hypo="You can come back anytime as our chat window is open 24/7"
    
#compute sentence level bleu
bleu = sacrebleu.BLEU(smooth_method="add-k",effective_order=True)
bleu_score=bleu.corpus_score(hypo, [ref])
