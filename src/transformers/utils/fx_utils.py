import ast
import copy
from ast import (
    NodeTransformer,
    Tuple,
    List,
    Constant,
    Load
)


class _IfStatementTracer(ast.NodeTransformer):

    def __init__(self, recording_tape_name: str):
        super().__init__()
        self.recording_tape_name = recording_tape_name
        self.record_mode = True
        self.tape = None

    def record(self):
        self.record_mode = True

    def transform(self, tape):
        if tape is None:
            raise ValueError("A recorded tape must be provided in transform mode")
        self.tape = tape
        self.record_mode = False

    def create_recorder_node(self, id_, value):
        expr = ast.parse(
            f"{self.recording_tape_name}[{id_}] = {value}", mode="exec"
        )
        return expr.body[0]

    def _visit_If_record_mode(self, if_):
        counter = 0
        to_visit = [if_]
        while to_visit:
            node = to_visit.pop(0)
            if isinstance(node, ast.If):
                node.body.insert(0, self.create_recorder_node(id(if_), counter))
                for child in node.body[1:]:
                    if not isinstance(child, ast.If):
                        continue
                    self._visit_If_record_mode(child)

                if node.orelse:
                    if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                        to_visit = node.orelse + to_visit
                    else:
                        to_visit = [node.orelse] + to_visit
            else:
                node.insert(0, self.create_recorder_node(id(if_), counter))
                # node is actually a set of nodes under the else statement.
                for child in node[1:]:
                    if not isinstance(child, ast.If):
                        continue
                    self._visit_If_record_mode(child)
            counter += 1
        return if_

    def _visit_If_transform_mode(self, if_):
        counter = 0
        to_visit = [if_]
        target = self.tape[id(if_)]
        node_to_return = None
        while to_visit:
            node = to_visit.pop(0)
            if isinstance(node, ast.If):
                if counter == target:
                    nodes = []
                    for child in node.body[1:]:
                        if not isinstance(child, ast.If):
                            nodes.append(child)
                        else:
                            nodes += self._visit_If_transform_mode(child)
                    node_to_return = nodes

                if node.orelse:
                    if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                        to_visit = node.orelse + to_visit
                    else:
                        to_visit = [node.orelse] + to_visit
            else:
                if counter == target:
                    # node is actually a set of nodes under the else statement.
                    nodes = []
                    for child in node[1:]:
                        if not isinstance(child, ast.If):
                            nodes.append(child)
                        else:
                            nodes += self._visit_If_transform_mode(child)
                    node_to_return = nodes
            counter += 1
        return node_to_return

    def visit_If(self, if_):
        if self.record_mode:
            return self._visit_If_record_mode(if_)
        else:
            res = self._visit_If_transform_mode(if_)
            self.tape = None
        return res



def resolve_name(name, d_or_o):
    names = name.split(".")
    res = d_or_o
    for n in names:
        if isinstance(res, dict):
            res = res.get(n)
        else:
            res = getattr(res, n)
        if res is None:
            raise AttributeError("blabla")
    return res


def trace_if_statements(source_ast, function_name, args=None, kwargs=None, globals_=None):
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    if globals_ is None:
        globals_ = {}
    source_ast = copy.deepcopy(source_ast)
    recording_tape_name = "traced_if_statements"
    if_tracer = _IfStatementTracer(recording_tape_name)

    # Recording.
    if_tracer.record()
    if_tracer.visit(source_ast)
    ast.fix_missing_locations(source_ast)
    code_for_recording = compile(source_ast, filename="<trace_if_statements>", mode="exec")
    cache = {recording_tape_name: {}, **globals_}
    exec(code_for_recording, cache)
    print(cache)
    transformed_func = resolve_name(function_name, cache)# cache[function_name]
    transformed_func(*args, **kwargs)
    recording_tape = cache[recording_tape_name]

    # Tracing.
    if_tracer.transform(recording_tape)
    if_tracer.visit(source_ast)
    ast.fix_missing_locations(source_ast)
    final_code = compile(source_ast, filename="<trace_if_statements>", mode="exec")
    final_cache = {**globals_}
    exec(final_code, final_cache)

    return final_cache[function_name]
