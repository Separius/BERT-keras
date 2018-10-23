class AbstractEncoder:
    def process(self, tokens, segment_ids, masks):
        raise NotImplementedError()
