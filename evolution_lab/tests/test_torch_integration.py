"""Optional real tensor tests; no downloaded model or network required."""
import tempfile
import unittest
from types import SimpleNamespace
try:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    from peft import LoraConfig, TaskType, get_peft_model, PeftModel
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from evolution_lab.experiment import fit, snapshot, restore, norm_matched_random
from evolution_lab.train_adapter import lora_delta_stats, supervised_batch


class Tokenizer:
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=False):
        text=''.join(m['role'][0]+':'+m['content']+'|' for m in messages)
        return text+('a:' if add_generation_prompt else '')
    def __call__(self,text,add_special_tokens=False,truncation=False,max_length=None,return_tensors=None):
        ids=[ord(c)%64 for c in text]
        if truncation: ids=ids[:max_length]
        if return_tensors:
            class Batch(dict):
                def to(self,device):
                    return Batch({k:v.to(device) for k,v in self.items()})
            return Batch(input_ids=torch.tensor([ids]),attention_mask=torch.ones(1,len(ids),dtype=torch.long))
        return {'input_ids':ids}


@unittest.skipUnless(ML_AVAILABLE,'Optional torch/transformers/peft not installed')
class TensorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.config=LlamaConfig(vocab_size=64,hidden_size=16,intermediate_size=32,
            num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2)
        self.base=LlamaForCausalLM(self.config).eval()
        self.inputs=torch.tensor([[1,2,3]])

    def test_training_random_control_and_reload(self):
        with torch.no_grad(): original=self.base(self.inputs).logits.clone()
        base_state={k:v.clone() for k,v in self.base.state_dict().items()}
        model=get_peft_model(self.base,LoraConfig(task_type=TaskType.CAUSAL_LM,r=2,
            lora_alpha=4,lora_dropout=0.,target_modules=['q_proj','v_proj'],bias='none')).eval()
        with torch.no_grad(): self.assertTrue(torch.equal(original,model(self.inputs).logits))
        frozen={k:v.clone() for k,v in model.named_parameters() if not v.requires_grad}
        local=SimpleNamespace(model=model,tokenizer=Tokenizer(),device='cpu')
        fit(local,[{'id':'one','prompt':'Hi','answer':'ok'}],3,.005,128,12)
        self.assertTrue(all(torch.equal(v,dict(model.named_parameters())[k]) for k,v in frozen.items()))
        trained=snapshot(model)
        delta=lora_delta_stats(model,torch)
        self.assertGreater(delta['delta_global_frobenius_norm'],0)
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            fresh=LlamaForCausalLM(self.config);fresh.load_state_dict(base_state)
            reloaded=PeftModel.from_pretrained(fresh,directory).eval()
            with torch.no_grad(): self.assertTrue(torch.allclose(model(self.inputs).logits,reloaded(self.inputs).logits))
        norm_matched_random(model,torch,100)
        random_delta=lora_delta_stats(model,torch)
        for a,b in zip(delta['layers'],random_delta['layers']):
            self.assertAlmostEqual(a['delta_frobenius_norm'],b['delta_frobenius_norm'],places=6)
        self.assertTrue(any(not torch.equal(v,snapshot(model)[k]) for k,v in trained.items()))
        restore(model,trained,torch)
        self.assertTrue(all(torch.equal(v,snapshot(model)[k]) for k,v in trained.items()))

    def test_answer_mask_and_truncation(self):
        batch=supervised_batch(Tokenizer(),{'id':'one','prompt':'Hi','answer':'ok'},128,'cpu')
        self.assertGreater((batch['labels']==-100).sum(),0)
        self.assertGreater((batch['labels']!=-100).sum(),0)
        with self.assertRaisesRegex(ValueError,'truncated'):
            supervised_batch(Tokenizer(),{'id':'one','prompt':'Hi','answer':'ok'},16,'cpu')

    def test_equal_mixture_is_exact_sum_of_effective_deltas(self):
        from evolution_lab.mixture import equal_mixture
        config=LoraConfig(task_type=TaskType.CAUSAL_LM,r=2,lora_alpha=4,
                          lora_dropout=0.,target_modules=['q_proj','v_proj'],bias='none')
        model=get_peft_model(self.base,config,adapter_name='first')
        model.add_adapter('second',config)
        with torch.no_grad():
            for module in model.modules():
                if hasattr(module,'lora_B'):
                    for layer in module.lora_B.values():layer.weight.normal_()
        originals={name:.5*module.get_delta_weight('first')+.5*module.get_delta_weight('second')
                   for name,module in model.named_modules() if hasattr(module,'lora_A')}
        equal_mixture(model,['first','second'])
        self.assertEqual(model.peft_config['mixture'].r,4)
        for name,module in model.named_modules():
            if name in originals:
                self.assertTrue(torch.allclose(module.get_delta_weight('mixture'),originals[name],atol=1e-6))
        mixture_delta=lora_delta_stats(model,torch,'mixture')
        self.assertEqual(mixture_delta['adapted_modules'],2)
        source_deltas={name:module.get_delta_weight('first').clone()
                       for name,module in model.named_modules() if hasattr(module,'lora_A')}
        norm_matched_random(model,torch,101,adapter_name='mixture')
        random_delta=lora_delta_stats(model,torch,'mixture')
        for left,right in zip(mixture_delta['layers'],random_delta['layers']):
            self.assertAlmostEqual(left['delta_frobenius_norm'],right['delta_frobenius_norm'],places=6)
        for name,module in model.named_modules():
            if name in source_deltas:
                self.assertTrue(torch.equal(module.get_delta_weight('first'),source_deltas[name]))

    def test_prompt_demonstrations_are_chat_turns(self):
        from evolution_lab.runner import LocalModel
        class RecordingTokenizer(Tokenizer):
            eos_token_id=0
            def apply_chat_template(self,messages,**kwargs):
                self.messages=messages
                return super().apply_chat_template(messages,**kwargs)
            def decode(self,tokens,**kwargs):return 'done'
        tokenizer=RecordingTokenizer()
        local=LocalModel.__new__(LocalModel)
        local.tokenizer=tokenizer;local.torch=torch;local.device='cpu'
        local.model=SimpleNamespace(generate=lambda **kw:torch.cat([kw['input_ids'],torch.tensor([[1]])],dim=1))
        response,tokens,_=local.generate('system','test',4,
            demonstrations=[{'id':'demo','prompt':'training prompt','answer':'training answer'}])
        self.assertEqual(response,'done');self.assertEqual(tokens,1)
        self.assertEqual([m['role'] for m in tokenizer.messages],['system','user','assistant','user'])
        self.assertEqual(tokenizer.messages[2]['content'],'training answer')
        local.generate('system','test',4)
        self.assertEqual([m['role'] for m in tokenizer.messages],['system','user'])
