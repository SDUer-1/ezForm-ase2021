var fs = require('fs');
var bodyParser = require('body-parser');
var express	= require("express");
var app	= express();

app.use(express.static("."));
app.use(bodyParser.urlencoded({ extended: true }));

app.get('/',function(req,res){
    res.sendfile("xml.html");
    let time = Date()
    console.log("Connecting at", time.toLocaleString())
});

app.post('/submitData', function (req, res) {
    let data = req.body
    console.log(data)
    fs.writeFile('data.json', JSON.stringify(data, null, '\t'), function (err) {
        if (! err){
            console.log('data store')
        }
        else{
            console.log(err)
        }
    });
})

app.listen(8000,function(){
    console.log("Working on port 8000");
});