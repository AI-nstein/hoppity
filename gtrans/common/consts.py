
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import argparse
import os
import numpy as np
import torch
import torch.nn as nn

t_float = torch.float32
np_float = np.float32
str_float = "float32"

SEPARATOR = '!#@$'
MAX_VOCAB_SIZE = 5000

opts = argparse.ArgumentParser(description='gpu option', allow_abbrev=False)
opts.add_argument('-gpu', type=int, default=-1, help='-1: cpu; 0 - ?: specific gpu index')

args, _ = opts.parse_known_args()
if torch.cuda.is_available() and args.gpu >= 0:
    DEVICE = torch.device('cuda:' + str(args.gpu))
    print('use gpu indexed: %d' % args.gpu)
else:
    DEVICE = torch.device('cpu')
    print('use cpu')


class Lambda(nn.Module):

    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


NONLINEARITIES = {
        "tanh": nn.Tanh(),
        "relu": nn.ReLU(),
        "softplus": nn.Softplus(),
        "sigmoid": nn.Sigmoid(),
        "elu": nn.ELU(),
        "swish": Swish(),
        "square": Lambda(lambda x: x**2),
        "identity": Lambda(lambda x: x),
        }


AST_EDGE_TYPE = 0
VAR_LINK_TYPE = 2
PREV_TOKEN_TYPE = 4

NUM_EDGE_TYPES = 6 # 3 edge types x 2 directions

USELESS_NODES = set(['COMMENT'])


CONTENT_NODE_TYPE = 'CONTENT_NODE'

ADDITIONAL_NODES = [CONTENT_NODE_TYPE]

OP_NONE = 'NoOp'
OP_ADD_NODE = 'add_node'
OP_DEL_NODE = 'del_node'
OP_REPLACE_VAL = 'replace_val'
OP_REPLACE_TYPE = 'replace_type'


js_keywords = ['break', 'case', 'catch', 'continue', 'debugger', 'default', 'delete', 'do', 'else', 'finally', 'for', 'function', 'if', 'in', 'instanceof', 'new', 'return', 'switch', 'this', 'throw', 'try', 'typeof', 'var', 'void', 'while', 'with', 'class', 'const', 'enum', 'export', 'extends', 'import', 'super', 'implements', 'interface', 'let', 'package', 'private', 'protected', 'public', 'static', 'yield']

HOPPITY_HOME = os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../setup.py')))

print('loading HOPPITY from', HOPPITY_HOME)

prefix = os.path.join(HOPPITY_HOME, "gtrans", "model")

shift_node_types = [
        "Program", 
        "Module", 
        "Script", 
        "Statement", 
        "IterationStatement", 
        "DoWhileStatement", 
        "ForInStatement", 
        "ForOfStatement", 
        "ForAwaitStatement", 
        "ForStatement", 
        "WhileStatement", 
        "ClassDeclaration", 
        "BlockStatement", 
        "BreakStatement", 
        "ContinueStatement", 
        "DebuggerStatement", 
        "EmptyStatement", 
        "ExpressionStatement", 
        "IfStatement", 
        "LabeledStatement", 
        "ReturnStatement", 
        "SwitchStatement", 
        "SwitchStatementWithDefault", 
        "ThrowStatement", 
        "TryCatchStatement", 
        "TryFinallyStatement", 
        "VariableDeclarationStatement", 
        "WithStatement", 
        "FunctionDeclaration", 
        "Expression", 
        "MemberExpression", 
        "ComputedMemberExpression", 
        "StaticMemberExpression", 
        "ClassExpression", 
        "LiteralBooleanExpression", 
        "LiteralInfinityExpression", 
        "LiteralNullExpression", 
        "LiteralNumericExpression", 
        "LiteralRegExpExpression", 
        "LiteralStringExpression", 
        "ArrayExpression", 
        "ArrowExpression", 
        "AssignmentExpression", 
        "BinaryExpression", 
        "CallExpression", 
        "CompoundAssignmentExpression", 
        "ConditionalExpression", 
        "FunctionExpression", 
        "IdentifierExpression", 
        "NewExpression", 
        "NewTargetExpression", 
"ObjectExpression", 
"UnaryExpression", 
"TemplateExpression", 
"ThisExpression", 
"UpdateExpression", 
"YieldExpression", 
"YieldGeneratorExpression", 
"AwaitExpression", 
"PropertyName", 
"ComputedPropertyName", 
"StaticPropertyName", 
"ObjectProperty", 
"NamedObjectProperty", 
"MethodDefinition", 
"Method", 
"Getter", 
"Setter", 
"DataProperty", 
"ShorthandProperty", 
"SpreadProperty", 
"ImportDeclaration", 
"Import", 
"ImportNamespace", 
"ExportDeclaration", 
"ExportAllFrom", 
"ExportFrom", 
"ExportLocals", 
"Export", 
"ExportDefault", 
"VariableReference", 
"BindingIdentifier", 
"AssignmentTargetIdentifier", 
"IdentifierExpression", 
"BindingWithDefault", 
"MemberAssignmentTarget", 
"ComputedMemberAssignmentTarget", 
"StaticMemberAssignmentTarget", 
"ArrayBinding", 
"ObjectBinding", 
"BindingProperty", 
"BindingPropertyIdentifier", 
"BindingPropertyProperty", 
"AssignmentTargetWithDefault", 
"ArrayAssignmentTarget", 
"ObjectAssignmentTarget", 
"AssignmentTargetProperty", 
"AssignmentTargetPropertyIdentifier", 
"AssignmentTargetPropertyProperty", 
"ClassElement", 
"ImportSpecifier", 
"ExportFromSpecifier", 
"ExportLocalSpecifier", 
"Block", 
"CatchClause", 
"Directive", 
"FormalParameters", 
"FunctionBody", 
"SpreadElement", 
"Super", 
"SwitchCase", 
"SwitchDefault", 
"TemplateElement", 
"VariableDeclaration", 
"VariableDeclarator" 
]


attr_order = {
"ArrayAssignmentTarget" :
['elements', 'rest'] ,
"ArrayBinding" :
['elements', 'rest'] ,
"ArrayExpression" :
['elements'] ,
"ArrowExpression" :
['isAsync', 'params', 'body'] ,
"AssignmentExpression" :
['binding', 'expression'] ,
"AssignmentTargetIdentifier" :
['name'] ,
"AssignmentTargetProperty" :
[] ,
"AssignmentTargetPropertyIdentifier" :
['binding', 'init'] ,
"AssignmentTargetPropertyProperty" :
['name', 'binding'] ,
"AssignmentTargetWithDefault" :
['binding', 'init'] ,
"AwaitExpression" :
['expression'] ,
"BinaryExpression" :
['left', 'operator', 'right'] ,
"BindingIdentifier" :
['name'] ,
"BindingProperty" :
[] ,
"BindingPropertyIdentifier" :
['binding', 'init'] ,
"BindingPropertyProperty" :
['name', 'binding'] ,
"BindingWithDefault" :
['binding', 'init'] ,
"Block" :
['statements'] ,
"BlockStatement" :
['block'] ,
"BreakStatement" :
['label'] ,
"CallExpression" :
['callee', 'arguments'] ,
"CatchClause" :
['binding', 'body'] ,
"Class" :
['super', 'elements'] ,
"ClassDeclaration" :
['name', 'super', 'elements'] ,
"ClassElement" :
['isStatic', 'method'] ,
"ClassExpression" :
['name', 'super', 'elements'] ,
"CompoundAssignmentExpression" :
['binding', 'operator', 'expression'] ,
"ComputedMemberAssignmentTarget" :
['object', 'expression'] ,
"ComputedMemberExpression" :
['object', 'expression'] ,
"ComputedPropertyName" :
['expression'] ,
"ConditionalExpression" :
['test', 'consequent', 'alternate'] ,
"ContinueStatement" :
['label'] ,
"DataProperty" :
['name', 'expression'] ,
"DebuggerStatement" :
[] ,
"Directive" :
['rawValue'] ,
"DoWhileStatement" :
['body', 'test'] ,
"EmptyStatement" :
[] ,
"Export" :
['declaration'] ,
"ExportAllFrom" :
['moduleSpecifier'] ,
"ExportDeclaration" :
[] ,
"ExportDefault" :
['body'] ,
"ExportFrom" :
['namedExports', 'moduleSpecifier'] ,
"ExportFromSpecifier" :
['name', 'exportedName'] ,
"ExportLocals" :
['namedExports'] ,
"ExportLocalSpecifier" :
['name', 'exportedName'] ,
"Expression" :
[] ,
"ExpressionStatement" :
['expression'] ,
"ForAwaitStatement" :
['left', 'right', 'body'] ,
"ForInStatement" :
['left', 'right', 'body'] ,
"FormalParameters" :
['items', 'rest'] ,
"ForOfStatement" :
['left', 'right', 'body'] ,
"ForStatement" :
['init', 'test', 'update', 'body'] ,
"Function" :
['isAsync', 'isGenerator', 'params', 'body'] ,
"FunctionBody" :
['directives', 'statements'] ,
"FunctionDeclaration" :
['isAsync', 'isGenerator', 'name', 'params', 'body'] ,
"FunctionExpression" :
['isAsync', 'isGenerator', 'name', 'params', 'body'] ,
"Getter" :
['name', 'body'] ,
"IdentifierExpression" :
['name'] ,
"IfStatement" :
['test', 'consequent', 'alternate'] ,
"Import" :
['defaultBinding', 'namedImports', 'moduleSpecifier'] ,
"ImportDeclaration" :
['moduleSpecifier'] ,
"ImportNamespace" :
['defaultBinding', 'namespaceBinding', 'moduleSpecifier'] ,
"ImportSpecifier" :
['name', 'binding'] ,
"IterationStatement" :
['body'] ,
"LabeledStatement" :
['label', 'body'] ,
"LiteralBooleanExpression" :
['value'] ,
"LiteralInfinityExpression" :
[] ,
"LiteralNullExpression" :
[] ,
"LiteralNumericExpression" :
['value'] ,
"LiteralRegExpExpression" :
['pattern', 'global', 'ignoreCase', 'multiLine', 'dotAll', 'unicode', 'sticky'] ,
"LiteralStringExpression" :
['value'] ,
"MemberAssignmentTarget" :
['object'] ,
"MemberExpression" :
['object'] ,
"Method" :
['isAsync', 'isGenerator', 'name', 'params', 'body'] ,
"MethodDefinition" :
['name', 'body'] ,
"Module" :
['directives', 'items'] ,
"NamedObjectProperty" :
['name'] ,
"NewExpression" :
['callee', 'arguments'] ,
"NewTargetExpression" :
[] ,
"Node" :
[] ,
"ObjectAssignmentTarget" :
['properties', 'rest'] ,
"ObjectBinding" :
['properties', 'rest'] ,
"ObjectExpression" :
['properties'] ,
"ObjectProperty" :
[] ,
"Program" :
[] ,
"PropertyName" :
[] ,
"ReturnStatement" :
['expression'] ,
"Script" :
['directives', 'statements'] ,
"Setter" :
['name', 'param', 'body'] ,
"ShorthandProperty" :
['name'] ,
"SpreadElement" :
['expression'] ,
"SpreadProperty" :
['expression'] ,
"Statement" :
[] ,
"StaticMemberAssignmentTarget" :
['object', 'property'] ,
"StaticMemberExpression" :
['object', 'property'] ,
"StaticPropertyName" :
['value'] ,
"Super" :
[] ,
"SwitchCase" :
['test', 'consequent'] ,
"SwitchDefault" :
['consequent'] ,
"SwitchStatement" :
['discriminant', 'cases'] ,
"SwitchStatementWithDefault" :
['discriminant', 'preDefaultCases', 'defaultCase', 'postDefaultCases'] ,
"TemplateElement" :
['rawValue'] ,
"TemplateExpression" :
['tag', 'elements'] ,
"ThisExpression" :
[] ,
"ThrowStatement" :
['expression'] ,
"TryCatchStatement" :
['body', 'catchClause'] ,
"TryFinallyStatement" :
['body', 'catchClause', 'finalizer'] ,
"UnaryExpression" :
['operator', 'operand'] ,
"UpdateExpression" :
['isPrefix', 'operator', 'operand'] ,
"VariableDeclaration" :
['kind', 'declarators'] ,
"VariableDeclarationStatement" :
['declaration'] ,
"VariableDeclarator" :
['binding', 'init'] ,
"VariableReference" :
['name'] ,
"WhileStatement" :
['test', 'body'] ,
"WithStatement" :
['object', 'body'] ,
"YieldExpression" :
['expression'] ,
"YieldGeneratorExpression" :
['expression']
}

