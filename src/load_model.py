
def load_text_lite(args, source_fp, device):
    from others.tokenization import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    sep_vid = tokenizer.vocab['[SEP]']
    cls_vid = tokenizer.vocab['[CLS]']
    n_lines = len(open(source_fp).read().split('\n'))

    def _process_src(raw):
        raw = raw.strip().lower()
        raw = raw.replace('[cls]','[CLS]').replace('[sep]','[SEP]')
        src_subtokens = tokenizer.tokenize(raw)
        src_subtokens = ['[CLS]'] + src_subtokens + ['[SEP]']
        src_subtoken_idxs = tokenizer.convert_tokens_to_ids(src_subtokens)
        src_subtoken_idxs = src_subtoken_idxs[:-1][:args.max_pos]
        src_subtoken_idxs[-1] = sep_vid
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        segs = segs[:args.max_pos]
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]

        src = torch.tensor(src_subtoken_idxs)[None, :].to(device)
        mask_src = (1 - (src == 0).float()).to(device)
        cls_ids = [[i for i, t in enumerate(src_subtoken_idxs) if t == cls_vid]]
        clss = torch.tensor(cls_ids).to(device)
        mask_cls = 1 - (clss == -1).float()
        clss[clss == -1] = 0

        return src, mask_src, segments_ids, clss, mask_cls

        with open(source_fp) as source:
            for x in tqdm(source, total=n_lines):
                src, mask_src, segments_ids, clss, mask_cls = _process_src(x)
                segs = torch.tensor(segments_ids)[None, :].to(device)
                batch = Batch()
                batch.src  = src
                batch.tgt  = None
                batch.mask_src  = mask_src
                batch.mask_tgt  = None
                batch.segs  = segs
                batch.src_str  =  [[sent.replace('[SEP]','').strip() for sent in x.split('[CLS]')]]
                batch.tgt_str  = ['']
                batch.clss  = clss
                batch.mask_cls  = mask_cls

                batch.batch_size=1
                yield batch
