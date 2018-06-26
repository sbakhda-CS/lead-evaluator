var request = require('request');

// Generic function for submitting ML training jobs
function main(params) {
    //console.log('params: ', params);

    var url = params.apiEndpoint + "/v2/jobs/" + params.properties.training_job_name + "/tasks";
    // console.log('url: ', url);
    
    var task = {
            payload: {
                command: ["--train", "--context", JSON.stringify(params) ]
            }
    };
    
    var options = {
        url : url,
        json: task,
        headers : {
            'Content-Type' : 'application/json',
            'Authorization' : 'Bearer ' + params.token
        }
    };

    return new Promise(function(resolve, reject) {
        request.post(options, function(error, response, body) {
            if (error) {
                reject({payload: error});
            }
            else {
                resolve({payload: body});
            }
        });
    });
}

module.exports = {main}
