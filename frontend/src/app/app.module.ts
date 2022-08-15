import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { DrawComponent } from './components/draw/draw.component';
import { UploadComponent } from './components/upload/upload.component';
import { OutputEditorComponent } from './components/output-editor/output-editor.component';
import { OutputFormatComponent } from './components/output-format/output-format.component';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { MaterialModule } from './material/material.module';
import { StatusComponent } from './components/status/status.component';
import { ResizeDirective } from './directives/resize.directive';

@NgModule({
  declarations: [
    AppComponent,
    DrawComponent,
    UploadComponent,
    OutputEditorComponent,
    OutputFormatComponent,
    StatusComponent,
    ResizeDirective
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    BrowserAnimationsModule,
    MaterialModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
