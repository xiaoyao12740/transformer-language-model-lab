import torch
from torch.utils.data import DataLoader

from src.dataset import HeldoutNextTokenDataset, contiguous_split, evenly_spaced_positions
from src.models import TransformerLanguageModel
from src.models.count_bigram import select_alpha
from src.tokenizer import CharacterTokenizer
from src.training.metrics import evaluate
from src.training.trainer import seed_everything


def fresh(seed):
    seed_everything(seed); return TransformerLanguageModel(8,4,8,2,1,0.0)


def test_same_seed_reproduces_fresh_model_and_step():
    a,b=fresh(42),fresh(42)
    assert all(torch.equal(x,y) for x,y in zip(a.state_dict().values(),b.state_dict().values(),strict=True))
    x=torch.tensor([[0,1,2,3]]); y=torch.tensor([[1,2,3,4]])
    for model in (a,b):
        opt=torch.optim.AdamW(model.parameters(),lr=.01); loss=model(x,y)[1]; opt.zero_grad(); loss.backward(); opt.step()
    assert all(torch.equal(x,y) for x,y in zip(a.state_dict().values(),b.state_dict().values(),strict=True))


def test_train_only_tokenizer_and_heldout_coverage():
    splits=contiguous_split("abc"*100); tok=CharacterTokenizer().fit(splits["train"])
    assert tok.decode(tok.encode(splits["val"]))==splits["val"] and tok.decode(tok.encode(splits["test"]))==splits["test"]


def test_validation_subset_spans_full_region():
    positions=evenly_spaced_positions(100,10,9)
    assert positions[0]==10 and positions[-1]==99 and len(positions)==len(set(positions))


def test_heldout_targets_counted_once():
    ds=HeldoutNextTokenDataset(list(range(20)),4)
    assert len(ds)==16 and [int(ds[i][1]) for i in range(len(ds))]==list(range(4,20))


def test_weighted_ce_with_uneven_batch():
    model=TransformerLanguageModel(5,2,8,2,1,0.0)
    ds=HeldoutNextTokenDataset([0,1,2,3,4,0,1],2)
    loader=DataLoader(ds,batch_size=3)
    measured=evaluate(model,loader,"cpu")["cross_entropy"]
    total=0
    for x,y in DataLoader(ds,batch_size=3): total+=torch.nn.functional.cross_entropy(model(x)[0][:,-1],y,reduction="sum").item()
    assert abs(measured-total/len(ds))<1e-7


def test_bigram_alpha_selection_uses_validation():
    train=[0,1,0,1,0,1,0,1]; val=[0,1,0,1,0,1]; positions=list(range(1,len(val)))
    alpha,_=select_alpha(train,val,positions,[.01,1.0])
    assert alpha==.01
