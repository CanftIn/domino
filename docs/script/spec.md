Prototype ::= def id '(' decl_list ')'

decl_list ::= Identifier | Identifier, decl_list

definition ::= Prototype Block

Block ::= { experssion_list }

experssion_list ::= block_expr ; experssion_list

block_expr ::= Decl | "return" | expr

Decl ::= var Identifier [ Type ] = expr

Type ::= < shape_list >

shape_list ::= number | number , shape_list

BinOpRHS ::= ('+' Primary)*

Expression ::= Primary BinOpRHS RHS

Primary ::= IdentifierExpr
         |  NumberExpr
         |  ParenExpr
         |  TensorLiteral

literalList ::= TensorLiteral
             |  TensorLiteral, literalList

TensorLiteral ::= [ literalList ] | number

NumberExpr ::= number

ParenExpr ::= '(' Expression ')'

IdentifierExpr ::= Identifier
                |  Identifier '(' Expression ')'

Return ::= "return" ';'
        |  "return" expr ';'