"""Pi0.5 RTX fixed graph contracts; no GPU operations are executed."""
import unittest
from types import SimpleNamespace as NS
import numpy as np
from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx

class FixedGraphTests(unittest.TestCase):
    def prompt_frontend(self, mode="fixed"):
        f=Pi05TorchFrontendRtx.__new__(Pi05TorchFrontendRtx)
        f._state_prompt_mode=mode; f._state_prompt_fixed_max_len=160
        f._batched_active=True;f._batch_size=2;f.max_prompt_len=160
        f.attn_backend=NS(per_slot_mask_enabled=True)
        f._normalize_state_for_prompt=lambda state:state
        f._embed_prompt_cached=lambda text, max_len, state:(np.zeros((7 if state is None else 45,2048),dtype=np.uint16),7 if state is None else 45)
        f._batch_graph_cache_size=8;f.batch_graph_bank_enabled=True
        f._batch_stream_cache={};f._frame_count=0
        captures=[]
        pipeline=NS(_graph=object(),set_language_embeds=lambda e:None,
                    set_language_embeds_batch=lambda e,valid_prompt_lens:captures.append((e,valid_prompt_lens)))
        f._batch_pipeline_cache={(2,160 if mode=="fixed" else 64):pipeline}
        return f,pipeline,captures

    def test_fixed_batch_reuses_same_graph_with_and_without_state(self):
        f,pipeline,captures=self.prompt_frontend()
        f.set_prompt_batch(['pick','pick'])
        f.set_prompt_batch(['pick','pick'],states=[np.zeros(8),np.ones(8)])
        self.assertIs(f.pipeline,pipeline)
        self.assertEqual(list(f._batch_pipeline_cache),[(2,160)])
        self.assertEqual([x.shape[0] for x in captures[0][0]],[160,160])
        self.assertEqual([x.shape[0] for x in captures[1][0]],[160,160])
        self.assertEqual(captures[0][1],[7,7])
        self.assertEqual(captures[1][1],[45,45])

    def test_fixed_capacity_overflow_fails_instead_of_truncating(self):
        f,_,_=self.prompt_frontend()
        f._embed_prompt_cached=lambda *a,**kw:(np.zeros((161,2048),dtype=np.uint16),161)
        with self.assertRaisesRegex(ValueError,'exceeds fixed capacity'):
            f.set_prompt_batch(['long','long'])

    def test_exact_batch_keeps_original_bucket_behavior(self):
        f,pipeline,captures=self.prompt_frontend(mode="exact")
        f._prompt_bucket_for=lambda length:64
        f.set_prompt_batch(['pick','pick'])
        self.assertIs(f.pipeline,pipeline)
        self.assertEqual(list(f._batch_pipeline_cache),[(2,64)])
        self.assertEqual([x.shape[0] for x in captures[0][0]],[64,64])

    def test_single_fixed_also_covers_no_state_prompt(self):
        f=Pi05TorchFrontendRtx.__new__(Pi05TorchFrontendRtx)
        f._rl_config=None;f._state_prompt_mode="fixed";f._state_prompt_fixed_max_len=160
        f._normalize_state_for_prompt=lambda state:state
        f._embed_prompt_cached=lambda *a,**kw:(np.zeros((7,2048),dtype=np.uint16),7)
        calls=[]
        f.pipeline=NS(_fixed_shape=True,set_language_embeds=lambda e:calls.append(e.shape[0]))
        f.attn_backend=NS(set_fixed_shape=lambda enabled:None)
        f._set_prompt_fixed=lambda length:calls.append(('fixed',length))
        f._set_prompt_per_length=lambda *a:self.fail('fixed mode must not select exact')
        f.set_prompt('pick',state=None)
        self.assertEqual(calls,[('fixed',7),7])

    def test_bank_keys_only_report_captured_graphs(self):
        f=Pi05TorchFrontendRtx.__new__(Pi05TorchFrontendRtx)
        f._batch_pipeline_cache={(2,160):NS(_graph=object()),(4,160):NS(_graph=None)}
        f._prompt_pipeline_cache={};f._fixed_pipeline=NS(_graph=object())
        f._state_prompt_fixed_max_len=160;f.pipeline=None
        self.assertEqual(f.batch_graph_bank_keys(),((1,160),(2,160)))

if __name__=='__main__':unittest.main()
