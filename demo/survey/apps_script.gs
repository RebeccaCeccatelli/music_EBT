// Receives survey submissions: appends them to the bound Google Sheet, emails
// each one to the study owner, and (if asked) emails the participant a copy.
// Setup: see demo/survey/README.md ("Collecting responses").
//
// The owner is the Google account the web app runs as ("Execute as: Me"), so
// no address needs to be written here. To send elsewhere, set OWNER_EMAIL.
var OWNER_EMAIL = "";

function doPost(e) {
  var lock = LockService.getScriptLock();
  lock.waitLock(10000);
  try {
    var d = JSON.parse(e.postData.contents);
    // The participant's address is only used to send their copy: never stored.
    var copyTo = d.copy_to || "";
    delete d.copy_to;
    var summary = d.summary || "";
    var body = JSON.stringify(d);

    var ss = SpreadsheetApp.getActiveSpreadsheet();
    var sheet = ss.getSheetByName("responses") || ss.insertSheet("responses");
    if (sheet.getLastRow() === 0) {
      sheet.appendRow(["received", "participant", "version", "n_trials", "training", "device", "json"]);
    }
    sheet.appendRow([
      new Date(), d.id, d.version, (d.responses || []).length,
      (d.background || {}).training || "", (d.background || {}).device || "", body,
    ]);

    var owner = OWNER_EMAIL || Session.getEffectiveUser().getEmail();
    MailApp.sendEmail({
      to: owner,
      subject: "Listening study: new response (" + d.id + ")",
      body: summary + "\n\n(Raw answers attached; also in the 'responses' sheet.)",
      attachments: [Utilities.newBlob(body, "application/json", "survey_" + d.id + ".json")],
    });
    if (copyTo && /^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(copyTo)) {
      MailApp.sendEmail({
        to: copyTo,
        subject: "Your answers: listening study on computer-made music",
        body: summary,
      });
    }
    return ContentService.createTextOutput("ok");
  } finally {
    lock.releaseLock();
  }
}
