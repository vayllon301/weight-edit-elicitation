"""The only inference adapter; never imported by freeze or report."""
import sys

import common as F


class Backend:
    def __init__(self):
        # Existing organism helpers also use the name common. Preserve our object,
        # then give the existing modules their own original import environment.
        sys.modules.pop('common', None)
        sys.path.insert(0, str(F.CODE / 'organisms'))
        sys.path.insert(0, str(F.CODE / 'oracle'))
        import common as O
        import editlib as E
        from mlx_lm import load
        self.O, self.E, self.load = O, E, load

    def loaded(self, spec, model):
        directory = spec['checkpoint']['directory'] if spec['kind'] == 'sft' else model['directory']
        obj, tok = self.load(directory)
        if spec['kind'] == 'saved':
            man = self.E.apply_edit_dir(obj, __import__('pathlib').Path(spec['edit']['directory']))
            F.require(F.tensor_set(man['tensors']) == spec['edit']['identity'], 'loaded edit identity differs')
        if spec['kind'] == 'noise':
            names = sorted(n for n in self.E.params(obj) if n.startswith('model.layers.'))
            tensors = self.E.noise_tensors(obj, spec['sigma'], spec['seed'], names)
            identity = F.tensor_set([{'name': n, 'edited_sha256': self.E.sha(t)} for n, t in tensors.items()])
            F.require(identity == spec['identity'], 'rebuilt noise differs from historical identity')
            self.E.set_tensors(obj, tensors)
        return obj, tok

    def check_noise(self, spec, model):
        obj, tok = self.loaded(spec, model)
        del obj, tok

    def score(self, spec, model, data, out, expected):
        obj, tok = self.loaded(spec, model)
        prefix, suffix = self.O.wrap(self.O.read_password()) if spec['kind'] == 'password' else ('', '')
        sanitizer = self.O.password_sanitizer()

        def sanitize(row):
            count = sanitizer(row)
            row['redacted'] = count > 0
            return count

        self.O.gsm8k_eval.run_eval(obj, tok, model_name=model['directory'], split=expected['split'],
                                 ids=[r['id'] for r in data], rows=data, out=out,
                                 prefix=prefix, suffix=suffix, history=spec.get('history'),
                                 sanitize=sanitize, extra_config=expected)
        del obj, tok
