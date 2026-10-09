// Receives survey submissions and appends them to the bound Google Sheet.
// Setup: see demo/survey/README.md ("Collecting responses").
function doPost(e) {
  var lock = LockService.getScriptLock();
  lock.waitLock(10000);
  try {
    var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName("responses")
      || SpreadsheetApp.getActiveSpreadsheet().insertSheet("responses");
    if (sheet.getLastRow() === 0) {
      sheet.appendRow(["received", "participant", "version", "n_trials", "training", "device", "json"]);
    }
    var body = e.postData.contents;
    var d = JSON.parse(body);
    sheet.appendRow([
      new Date(), d.id, d.version, (d.responses || []).length,
      (d.background || {}).training || "", (d.background || {}).device || "", body,
    ]);
    return ContentService.createTextOutput("ok");
  } finally {
    lock.releaseLock();
  }
}
