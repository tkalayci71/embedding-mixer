# Embedding Mixer version 0.11
#
# https://github.com/tkalayci71/embedding-mixer
#

import gradio as gr
from modules.script_callbacks import on_ui_tabs
from os import path
from modules import sd_hijack, shared
from matplotlib import pyplot as plt
from modules.textual_inversion.textual_inversion import Embedding
from io import BytesIO
from PIL import Image
import torch, math, random

#-------------------------------------------------------------------------------

def get_data():

    loaded_embs = sd_hijack.model_hijack.embedding_db.word_embeddings

    embedder = shared.sd_model.cond_stage_model.wrapped
    if embedder.__class__.__name__=='FrozenCLIPEmbedder': # SD1.x detected
        tokenizer = embedder.tokenizer
        internal_embs = embedder.transformer.text_model.embeddings.token_embedding.wrapped.weight

    elif embedder.__class__.__name__=='FrozenOpenCLIPEmbedder': # SD2.0 detected
        from modules.sd_hijack_open_clip import tokenizer as open_clip_tokenizer
        tokenizer = open_clip_tokenizer
        internal_embs = embedder.model.token_embedding.wrapped.weight

    else:
        tokenizer = None
        internal_embs = None

    return tokenizer, internal_embs, loaded_embs

#-------------------------------------------------------------------------------

def text_to_emb_ids(text, tokenizer):

    text = text.lower()

    if tokenizer.__class__.__name__== 'CLIPTokenizer': # SD1.x detected
        emb_ids = tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]

    elif tokenizer.__class__.__name__== 'SimpleTokenizer': # SD2.0 detected
        emb_ids =  tokenizer.encode(text)

    else:
        emb_ids = None

    return emb_ids # return list of embedding IDs for text

#-------------------------------------------------------------------------------

def emb_id_to_name(emb_id, tokenizer):

    emb_name_utf8 = tokenizer.decoder.get(emb_id)

    if emb_name_utf8 != None:
        byte_array_utf8 = bytearray([tokenizer.byte_decoder[c] for c in emb_name_utf8])
        emb_name = byte_array_utf8.decode("utf-8", errors='backslashreplace')
    else:
        emb_name = '!Unknown ID!'

    return emb_name # return embedding name for embedding ID

#-------------------------------------------------------------------------------

def get_embedding_info(text):

    text = text.strip().lower()

    tokenizer, internal_embs, loaded_embs = get_data()

    loaded_emb = loaded_embs.get(text, None)

    if loaded_emb == None:
        for k in loaded_embs.keys():
            if text == k.lower():
                loaded_emb = loaded_embs.get(k, None)
                break

    if loaded_emb!=None:
        emb_name = loaded_emb.name
        emb_id = '['+loaded_emb.checksum()+']' # emb_id is string for loaded embeddings
        emb_vec = loaded_emb.vec
        return emb_name, emb_id, emb_vec, loaded_emb #also return loaded_emb reference

    # support for #nnnnn format
    val = None
    if text.startswith('#'):
        try:
            val = int(text[1:])
            if (val<0) or (val>=internal_embs.shape[0]): val = None
        except:
            val = None

    # obtain internal embedding ID
    if val!=None:
        emb_id = val
    else:
        emb_ids = text_to_emb_ids(text, tokenizer)
        if len(emb_ids)==0: return None, None, None, None
        emb_id = emb_ids[0] # emb_id is int for internal embeddings

    emb_name = emb_id_to_name(emb_id, tokenizer)
    emb_vec = internal_embs[emb_id].unsqueeze(0)

    return emb_name, emb_id, emb_vec, None # return embedding name, ID, vector

#-------------------------------------------------------------------------------

def shape_str(vec):
    return str(vec.shape[0])+'x'+str(vec.shape[1])

def emb(name):
    emb_name, emb_id, emb_vec, loaded_emb = get_embedding_info(name)
    emb_id_str = str(emb_id)
    if not emb_id_str.startswith('['): emb_id_str = '#'+emb_id_str
    formula_log.append('emb("'+name+'") = '+emb_name+' '+emb_id_str+', size is '+shape_str(emb_vec))
    return emb_vec.to(device='cpu',dtype=torch.float32)

def concat(*args):
    tot_vec = torch.concat(args)
    formula_log.append('concat '+str(len(args))+' vectors, result is '+shape_str(tot_vec))
    return tot_vec

def mix(*args):
    vec_size = args[0].shape[1]
    tot_vec = torch.zeros(vec_size).unsqueeze(0)
    for mix_vec in args:
        padding = torch.zeros(abs(tot_vec.shape[0]-mix_vec.shape[0]),vec_size)
        if mix_vec.shape[0]<tot_vec.shape[0]:
            mix_vec = torch.cat([mix_vec, padding])
        else:
            tot_vec = torch.cat([tot_vec, padding])
        tot_vec+= mix_vec
    formula_log.append('mix '+str(len(args))+' vectors, result is '+shape_str(tot_vec))
    return tot_vec

def reduce(vec):
    tot_vec = torch.sum(vec,dim=0,keepdim=True)
    formula_log.append('reduce '+shape_str(vec)+' to 1-vector, result is '+shape_str(tot_vec))
    return tot_vec

def extract(source_vec,vec_nos):
    tot_vec = None
    vec_count = int(source_vec.shape[0])
    for i in range(vec_count):
        if (i in vec_nos):
            if tot_vec==None:
                tot_vec = source_vec[i].unsqueeze(0)
            else:
                tot_vec = torch.concat([tot_vec, source_vec[i].unsqueeze(0)])
    formula_log.append('extract vectors '+str(vec_nos)+' from '+shape_str(source_vec)+', result is '+shape_str(tot_vec))
    return tot_vec

def remove(source_vec,vec_nos):
    tot_vec = None
    vec_count = int(source_vec.shape[0])
    for i in range(vec_count):
        if not (i in vec_nos):
            if tot_vec==None:
                tot_vec = source_vec[i].unsqueeze(0)
            else:
                tot_vec = torch.concat([tot_vec, source_vec[i].unsqueeze(0)])
    formula_log.append('remove vectors '+str(vec_nos)+' from '+shape_str(source_vec)+', result is '+shape_str(tot_vec))
    return tot_vec

def process(source_vec, eval_txt):
    tot_vec = source_vec
    vec = tot_vec.clone()
    try:
        maxn = vec.shape[0]
        maxi = vec.shape[1]
        for n in range(maxn):

            vec_mag = torch.linalg.norm(vec[n])
            vec_min = torch.min(vec[n])
            vec_max = torch.max(vec[n])

            if eval_txt.startswith('='):
                #item-wise eval
                for i in range(maxi):
                    v = vec[n,i]
                    ve = eval(eval_txt[1:]) #strip "="
                    vec[n,i] = ve
            else:
                #tensor-wise eval
                v = vec[n]
                ve = eval(eval_txt)
                vec[n] = ve
        tot_vec = vec
        formula_log.append('processed "'+eval_txt+'" on '+shape_str(tot_vec))
    except Exception as e:
        formula_log.append('Error processing: "'+eval_txt+'" - '+str(e))
    return tot_vec

#-------------------------------------------------------------------------------

def figure_to_image(fig):
    buf = BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    img.load()
    buf.close()
    return img

#-------------------------------------------------------------------------------

def eval_formula(formula_str):
    tot_vec = None

    log = []

    try:
        global formula_log
        formula_log = []
        tot_vec = eval(formula_str)
        log.append('\n'.join(formula_log))
    except Exception as e:
        log.append('\n'.join(formula_log))
        log.append('Error evaluating formula: '+str(e))
        tot_vec = None

    valid = False
    if (tot_vec!=None):
        if type(tot_vec)!=torch.Tensor:
            log.append('Result type must be torch.Tensor, not '+str(type(tot_vec)))
        else:
            if len(tot_vec.shape)==1: tot_vec =tot_vec.unsqueeze(0)
            if (len(tot_vec.shape)!=2) or ((tot_vec.shape[1]!=768) and (tot_vec.shape[1]!=1024)):
                log.append('Tensor shape is invalid: '+str(tot_vec.shape))
            else:
                valid = True

    if valid==False: tot_vec = None

    return tot_vec, '\n'.join(log)

#-------------------------------------------------------------------------------

def do_save(step_str, formula_str, save_name, enable_overwrite):

    step_str = step_str.strip().lower()
    save_name = save_name.strip().lower()
    formula_str = formula_str.strip()
    if (formula_str==''): return 'Error: Formula is empty', None

    log = []

    save_filename = 'embeddings/'+save_name+'.bin'
    file_exists = path.exists(save_filename)
    if (file_exists):
        if not(enable_overwrite):
            return 'Error: File already exists ('+save_filename+'), overwrite is not enabled, aborting', None
        else:
            log.append('File already exists, overwrite is enabled')

    saved_graph = None
    tot_vec = None

    tot_vec, eval_log = eval_formula(formula_str)
    if eval_log!='': log.append(eval_log)

    if (tot_vec==None):
        log.append('Warning: Nothing to save')
    elif save_name=='':
         log.append('Error: Filename is empty')
    else:
        new_emb = Embedding(tot_vec, save_name)

        if step_str!='':
            try:
                step_val = int(step_str)
                new_emb.step = step_val
                log.append('Setting step value to '+str(step_val))
            except:
                log.append('Warning: Step value is invalid, ignoring')

        try:
            new_emb.save(save_filename)
            log.append('Saved "'+save_filename+'"')
        except:
            log.append('Error saving "'+save_filename+'" (filename might be invalid)')

        log.append('Reloading all embeddings')
        sd_hijack.model_hijack.embedding_db.dir_mtime=0
        sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()

        fig = plt.figure()
        for u in range(tot_vec.shape[0]):
            x = torch.arange(start=0, end=tot_vec[u].shape[0], step=1)
            plt.plot(x.numpy(), tot_vec[u].numpy())
            saved_graph = figure_to_image(fig)

    return '\n'.join(log), saved_graph

#-------------------------------------------------------------------------------

def add_tab():
    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Row():
            formula_str = gr.Textbox(label="Formula", lines=12)
        with gr.Row():
            with gr.Column(scale=1): step_str = gr.Textbox(label="Step", lines=1, placeholder='only for training')
            with gr.Column(scale=1): save_name = gr.Textbox(label="Filename", lines=1, placeholder='Enter file name to save')
            with gr.Column(scale=1): save_button = gr.Button(value="Save embedding", variant="primary")
            with gr.Column(scale=1): enable_overwrite = gr.Checkbox(label="Enable overwrite", value=False)
        with gr.Row():
            with gr.Column(scale=2): save_log = gr.Textbox(label="Save log", lines=10)
            with gr.Column(scale=1): save_graph = gr.Image(label="Save graph", interactive=False)

        save_button.click(fn=do_save, inputs = [step_str, formula_str, save_name, enable_overwrite], outputs =[save_log, save_graph])

    return [(ui, "Embedding Mixer", "Mixer")]

on_ui_tabs(add_tab)
