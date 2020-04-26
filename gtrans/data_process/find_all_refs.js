const tsMorph = require("ts-morph");
const ts = tsMorph.ts;
const fs = require("fs")
const {parseScriptWithLocation, parseModuleWithLocation} = require("shift-parser");
const { spawnSync } = require('child_process');

var loc_to_node_num = {}
var ref_map = {}

function get_shift_path_from_num(shift_node_num, node, path=""){

	if (parseInt(shift_node_num) === node["idx"]){
		return path
	}else{
		let prev_path = path

		for(var key in node){
			path = prev_path
			let val = node[key];

			let p = undefined;
			if(val instanceof Array){
				let count = 0
				try{
					val.forEach(function(entry){
						if(entry){
							path = prev_path
							path += key + "/" + count.toString() + "/"
							p = get_shift_path_from_num(shift_node_num, entry, path);
							if(p) throw "Break";
						}

						count += 1
					});
				} catch(e) {
				}

			}else if(val instanceof Object){
				path += key + "/"
				p = get_shift_path_from_num(shift_node_num, val, path);

			}

			if(p) {
				return p;
			}
		}

	}

}

function get_ast_num(shift_node_num, shift_root, value){
	let path = get_shift_path_from_num(shift_node_num, shift_root)

	if(!path){
		console.log("no path", shift_node_num)
		process.exit(1)
	}
	return path_to_ast_num(ast, path, value)
}

function path_to_ast_num(ast_root, path, value){

	path = path.split("/")
	path = path.filter(function(elem){
		return elem !== "" && elem !== " "
	})

	let node = ast_root
	path.forEach(function(path_step){
		let count = -1;
		if(!isNaN(path_step)){ //is a number
			count = path_step
		}

		let inner_count = 0;
		let found = false
		node["children"].forEach(function(child){
			if(child["node_type"].includes(path_step)){
				if(count < 0 || inner_count === count ){
					node = child
					found = true
				}else{
					inner_count += 1;
				}
			}
		});

		if(!found && count >= 0){
			node = node["children"][count] 
			found = true
		}

		if(!found){
			console.log("KEY ERROR", node, path_step)
			console.log(path)
			process.exit(1)
		}
	});


	//console.log(node, value)
	if(node["value"] === value){
		return node.index;
	}

	let filtered = node["children"].filter(function(child){
		return child["value"] === value;
	})


	
	if(filtered.length > 1){
		console.log("MORE THAN ONE CHILD OPTION")
		process.exit(1)
	}

	if(filtered.length > 0){
		return filtered[0]["index"]
	}

	return node.index;
}


//returns true is pos1 comes earlier in the source code than pos2
function lt(pos1, pos2){
	return pos1["line"] < pos2["line"] || 
		pos1["line"] === pos2["line"] && pos1["column"] <= pos2["column"]
}


var count = 0
function build_loc_map(node){
	if(!node) return

	node.idx = count;

	let loc_key = {}
	if(locations.get(node)){
		let loc = locations.get(node).start
		let end = locations.get(node).end
		loc_key["start"] = {}
		loc_key["end"] = {}
		loc_key["start"]["line"] = loc["line"]
		loc_key["start"]["column"] = loc["column"]+1
		loc_key["end"]["line"] = end["line"]
		loc_key["end"]["column"] = end["column"]+1
		loc_to_node_num[JSON.stringify(loc_key)] = count;
	}

	for(var key in node){
		let val = node[key];

		if(val instanceof Array){
			val.forEach(function(entry){
				count += 1
				build_loc_map(entry);
			});

		}else if(val instanceof Object){
			count += 1
			build_loc_map(val);
		}
	}
}

function get_nearest_js_doc(node){
	let jsDoc = undefined
	while(node.parent && !jsDoc){
		if(node.jsDoc){
			jsDoc = node.jsDoc
		}else{
			node = node.parent;
		}

	}

	return jsDoc;
}

function get_node_num(start_pos, end_pos){

	let start = sourceFile.getLineAndColumnAtPos(start_pos);
	let end = sourceFile.getLineAndColumnAtPos(end_pos);

	for(var dict_loc in loc_to_node_num){
		let key = JSON.parse(dict_loc)
		if(start["line"] === key["start"]["line"] && start["column"] === key["start"]["column"] &&
			end["line"] === key["end"]["line"] && end["column"] === key["end"]["column"]){
			return loc_to_node_num[dict_loc]
		}
	}

	let ret = undefined;
	for(var dict_loc in loc_to_node_num){
		let key = JSON.parse(dict_loc)
		if((lt(key["start"], start) && end["line"] === key["end"]["line"] && end["column"] === key["end"]["column"]) &&
		   (!ret || lt(ret["start"], key["start"]))){
			ret = key
		}

	}

	return loc_to_node_num[JSON.stringify(ret)]
}

function get_start_pos(node){
	return node.getPos() + node.getLeadingTriviaWidth()
}

function visit(node){
	try{
		if(node.getKind() === ts.SyntaxKind.Identifier){
			let refs = node.getDefinitionNodes();
			let node_pos = sourceFile.getLineAndColumnAtPos(get_start_pos(node));

			let key = get_node_num(get_start_pos(node), node.getEnd())

			if(!key){
				console.log("couldn't find a node")
				process.exit(1)
			}


			let ast_key = get_ast_num(key, tree, node.getText())
			ref_map[ast_key] = []


			refs.forEach(function(ref){
				let ref_pos = sourceFile.getLineAndColumnAtPos(get_start_pos(ref));
				if(JSON.stringify(ref_pos) !== JSON.stringify(node_pos)){

					let node_num = get_node_num(get_start_pos(ref), ref.getEnd())

					if(!node_num){
						let jsDoc = get_nearest_js_doc(node.compilerNode)
						if(jsDoc && jsDoc[0].pos < ref.getPos() && jsDoc[0].end > ref.getEnd()){
							return
						}
						

						//webpack line no bug 
						return
				
					}

					let ast_num = get_ast_num(node_num, tree, node.getText())

					ref_map[ast_key].push(ast_num);
				}
			}); 

		}
	}catch(e){
		//console.log(e)
	}
}

const src_file = process.argv[2];
const ast_file = process.argv[3]

const HOPPITY_HOME = __dirname
var ast = spawnSync("python", [HOPPITY_HOME + "/depickle.py", ast_file])
ast = ast.stdout.toString().split("\n")[3]

ast = JSON.parse(ast)

const project = new tsMorph.Project({
	compilerOptions: {
		allowJs: true
	}
});

const sourceFile = project.addSourceFileAtPath(src_file);
const src = fs.readFileSync(src_file, "utf-8")

let {tree, locations, comments} = parseModuleWithLocation(src);

build_loc_map(tree);
sourceFile.forEachDescendant(node => visit(node))
console.log(JSON.stringify(ref_map))

