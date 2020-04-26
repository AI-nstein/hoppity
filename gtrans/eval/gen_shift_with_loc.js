const fs = require("fs")
const {parseScriptWithLocation, parseModuleWithLocation} = require("shift-parser");

if (process.argv.length < 4){
	console.log("please specify src file and at least one path")
	process.exit(1)
}

const src_file = process.argv[2]

let paths = []
for(let i =3; i<process.argv.length; i++){
	paths.push(process.argv[i]);
}


src = fs.readFileSync(src_file, "utf-8")
let {tree, locations, comments} = parseModuleWithLocation(src);

let locs = []

paths.forEach(function(path){
	fields = path.split("/")

	let prev_prop = []
	prop = tree
	for (let i=0; i<fields.length; i++){
		let field = fields[i];
		prev_prop.push(prop)

		if(field !== ""){
			prop = prop[field]
		}
	}


	loc = locations.get(prop);

	while (!loc) {
		prop = prev_prop.pop()
		loc = locations.get(prop);
	}

	locs.push(loc)
});

console.log(JSON.stringify(locs));
